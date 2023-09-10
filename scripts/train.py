import os
import sys
import json
import h5py
import argparse
import importlib
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import tensorflow as tf

from torch.utils.data import DataLoader
from datetime import datetime
from copy import deepcopy

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from lib.data_util import get_dataloader
from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.dataset import ScannetReferenceDataset
from lib.solver import Solver
from lib.config import CONF
from models.refnet import RefNet
from models.Object_Detection import Object_Detection

from scripts.utils.AdamW import AdamW
from scripts.utils.script_utils import set_params_lr_dict

DC = ScannetDatasetConfig()

def get_model(args):
    # initiate model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(not args.no_height)
    model = RefNet(
        num_class=DC.num_class,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,
        args = args,
        input_feature_dim=input_channels,
        num_proposal=args.num_proposals,
        use_lang_classifier=(not args.no_lang_cls),
        use_bidir=args.use_bidir,
        no_reference=args.no_reference,
        chunking = args.use_chunking,
        lang_module = args.lang_module,
    )

    # trainable model
    if args.use_pretrained:
        # load model
        if args.detection_module == "votenet":
            print("loading pretrained VoteNet...")
        elif args.detection_module == "3detr":
            print("loading pretrained 3DETRm...")
        pretrained_model = RefNet(
            num_class=DC.num_class,
            num_heading_bin=DC.num_heading_bin,
            num_size_cluster=DC.num_size_cluster,
            mean_size_arr=DC.mean_size_arr,
            args = args,
            num_proposal=args.num_proposals,
            input_feature_dim=input_channels,
            use_bidir=args.use_bidir,
            no_reference=False,
            chunking = args.use_chunking,
            lang_module = args.lang_module,
        )

        if args.detection_module == "votenet":
        
            pretrained_path = os.path.join(CONF.PATH.OUTPUT, args.use_pretrained, "model.pth")
            pretrained_model.load_state_dict(torch.load(pretrained_path), strict=False)
            # mount
            model.backbone_net = pretrained_model.backbone_net
            model.vgen = pretrained_model.vgen
            model.proposal = pretrained_model.proposal
            
            if args.no_detection:
                # freeze pointnet++ backbone
                for param in model.backbone_net.parameters():
                    param.requires_grad = False
    
                # freeze voting
                for param in model.vgen.parameters():
                    param.requires_grad = False
                
                # freeze detector
                for param in model.proposal.parameters():
                    param.requires_grad = False
        
        elif args.detection_module == "3detr": 
            # 3DETR pretrained:            
            for param in pretrained_model.parameters():
                for weights in param.data:
                    print(weights)
                    break
                break
            
            pretrained_path = os.path.join(CONF.PATH.OUTPUT, args.use_pretrained, "model_last.pth")
            pre = torch.load(pretrained_path)
    
            pretrained_model.load_state_dict(torch.load(pretrained_path), strict=False)
    
            for key in pre:
                sd = pretrained_model.state_dict()
                sd[key[17:]] = pre[key]
                pretrained_model.load_state_dict(sd, strict=False)
        
            # mount
            
            for param in pretrained_model.parameters():
                for weights in param.data:
                    print(weights)
                    break
                break
            
            model.Object_Detection = pretrained_model.Object_Detection
            #print(model)
            
            if args.no_detection:
                # freeze 3DETR
                for param in model.Object_Detection.parameters():
                    param.requires_grad = False
                    

    print(model)
    model = model.cuda()

    return model

def get_optimizer(args, model):
    # Optimizer for 3DETR
    params_with_decay = []
    params_without_decay = []
    for name, param in model.named_parameters():
        if param.requires_grad is False:
            continue
        if args.filter_biases_wd and (len(param.shape) == 1 or name.endswith("bias")):
            params_without_decay.append(param)
        else:
            params_with_decay.append(param)

    if args.filter_biases_wd:
        param_groups = [
            {"params": params_without_decay, "weight_decay": 0.0},
            {"params": params_with_decay, "weight_decay": args.weight_decay},
        ]
    else:
        param_groups = [
            {"params": params_with_decay, "weight_decay": args.weight_decay},
        ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.detr_lr)
    return optimizer

def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params

def get_solver(args, dataloader):
    model = get_model(args)
    
    if args.detection_module == "votenet":
        optimizer_main = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        optimizer_det = None
        optimizer_lang = None
        optimizer_match = None
    # for 3DETR:
    
    elif args.detection_module == "3detr" and args.no_reference:
        optimizer_main = get_optimizer(args, model)
        optimizer_det = None
        optimizer_lang = None
        optimizer_match = None

    elif args.detection_module=="3detr" and args.lang_module=="bert" and args.match_module =="dvg" and args.sep_optim:
        # det optim
        detr_params = sum(p.numel() for p in model.Object_Detection.parameters())
        print(str(detr_params/1000000) + " mil. parameters in Detection module")
        optimizer_det = get_optimizer(args, model.Object_Detection)
        # lang optim
        lang_params = list(model.lang_encoder.parameters())
        optimizer_lang = optim.AdamW(lang_params, lr=args.lr_bert, weight_decay=args.bert_wd)
        l_params = sum(p.numel() for p in model.lang_encoder.parameters())
        print(str(l_params/1000000) + " mil. parameters in Language module")
        #dvg optim
        match_params = list(model.match.parameters())
        optimizer_match = optim.AdamW(match_params, lr=args.lr_match, weight_decay=args.match_wd)
        l_params = sum(p.numel() for p in model.match.parameters())
        print(str(l_params/1000000) + " mil. parameters in Match module")
        # rest
        rest_params = list(model.Object_Feature_MLP.parameters())
        other_params = sum(p.numel() for p in rest_params)
        print(str(other_params/1000000) + " mil. parameters for other modules")
        optimizer_main = optim.Adam(rest_params, lr=args.lr, weight_decay=args.wd)

    elif args.detection_module=="3detr" and args.lang_module=="bert" and args.sep_optim:
        # 3detr optim
        detr_params = sum(p.numel() for p in model.Object_Detection.parameters())
        print(str(detr_params/1000000) + " mil. parameters in Detection module")
        optimizer_det = get_optimizer(args, model.Object_Detection)
        # bert optim
        lang_params = list(model.lang_encoder.parameters())
        optimizer_lang = optim.AdamW(lang_params, lr=args.lr_bert, weight_decay=args.bert_wd)
        l_params = sum(p.numel() for p in model.lang_encoder.parameters())
        print(str(l_params/1000000) + " mil. parameters in Language module")
        # rest optim
        rest_params = list(model.Object_Feature_MLP.parameters()) + list(model.match.parameters())
        other_params = sum(p.numel() for p in rest_params)
        print(str(other_params/1000000) + " mil. parameters for other modules")
        optimizer_main = optim.Adam(rest_params, lr=args.lr, weight_decay=args.wd)
        optimizer_match = None

    elif args.detection_module == "3detr" and args.match_module == "scanrefer" and args.sep_optim:
        detr_params = sum(p.numel() for p in model.Object_Detection.parameters())
        print(str(detr_params/1000000) + " mil. parameters in Detection module")
        optimizer_det = get_optimizer(args, model.Object_Detection)
        rest_params = list(model.Object_Feature_MLP.parameters()) + list(model.lang_encoder.parameters())+ list(model.match.parameters())
        other_params = sum(p.numel() for p in rest_params)
        print(str(other_params/1000000) + " mil. parameters for other modules")
        optimizer_main = optim.Adam(rest_params, lr=args.lr, weight_decay=args.wd)
        optimizer_lang = None
        optimizer_match = None
    elif args.match_module != "scanrefer" and args.sep_optim:
        # 3detr param
        detr_params = sum(p.numel() for p in model.Object_Detection.parameters())
        print(str(detr_params/1000000) + " mil. parameters in Detection module")
        # dvg optim
        match_params = list(model.match.parameters())
        optimizer_match = optim.AdamW(match_params, lr=args.lr_match, weight_decay=args.match_wd)
        l_params = sum(p.numel() for p in model.match.parameters())
        print(str(l_params/1000000) + " mil. parameters in Match module")
        # rest optim
        rest_params = list(model.lang_encoder.parameters())# + list(model.Object_Feature_MLP.parameters())
        other_params = sum(p.numel() for p in rest_params)
        print(str(other_params/1000000) + " mil. parameters for other modules")
        optimizer_main = optim.Adam(rest_params, lr=args.lr, weight_decay=args.wd)
        optimizer_lang = None
        optimizer_det = None
    else:
        optimizer_main = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        optimizer_det = None
        optimizer_lang = None
        optimizer_match = None

    if args.use_checkpoint:
        print("loading checkpoint {}...".format(args.use_checkpoint))
        stamp = args.use_checkpoint
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        checkpoint = torch.load(os.path.join(CONF.PATH.OUTPUT, args.use_checkpoint, "checkpoint.tar"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer_main.load_state_dict(checkpoint["optimizer_main_state_dict"])
        if args.detection_module=="3detr" and args.sep_optim:
            optimizer_det.load_state_dict(checkpoint["optimizer_det_state_dict"])
        if args.lang_module=="bert" and args.sep_optim:
            optimizer_lang.load_state_dict(checkpoint["optimizer_lang_state_dict"])
        if args.lang_module=="dvg" and args.sep_optim:
            optimizer_match.load_state_dict(checkpoint["optimizer_match_state_dict"])
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.tag: stamp += "_"+args.tag.upper()
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        os.makedirs(root, exist_ok=True)

    # scheduler parameters for training solely the detection pipeline
    LR_DECAY_STEP = [80, 120, 160] if args.no_reference else None
    LR_DECAY_RATE = 0.1 if args.no_reference else None
    BN_DECAY_STEP = 20 if args.no_reference else None
    BN_DECAY_RATE = 0.5 if args.no_reference else None



    solver = Solver(
        model=model, 
        config=DC,
        args=args,
        dataloader=dataloader, 
        optimizer_main=optimizer_main, 
        optimizer_det=optimizer_det,
        optimizer_lang=optimizer_lang,
        optimizer_match=optimizer_match,
        stamp=stamp, 
        val_step=args.val_step,
        detection=not args.no_detection,
        reference=not args.no_reference, 
        use_lang_classifier=not args.no_lang_cls,
        lr_decay_step=LR_DECAY_STEP,
        lr_decay_rate=LR_DECAY_RATE,
        bn_decay_step=BN_DECAY_STEP,
        bn_decay_rate=BN_DECAY_RATE,
        detection_module=args.detection_module
    )
    num_params = get_num_params(model)

    return solver, num_params, root

def save_info(args, root, num_params, train_dataset, val_dataset):
    info = {}
    for key, value in vars(args).items():
        info[key] = value
    
    info["num_train"] = len(train_dataset)
    info["num_val"] = len(val_dataset)
    info["num_train_scenes"] = len(train_dataset.scene_list)
    info["num_val_scenes"] = len(val_dataset.scene_list)
    info["num_params"] = num_params

    with open(os.path.join(root, "info.json"), "w") as f:
        json.dump(info, f, indent=4)


def write_string_summary_v2(writer, s):
    with writer.as_default():
        tf.summary.text('Model configuration', s, step=0)

# Get model summary as a string
def get_summary_str(args, weight_dict):
    lines = []
    string = ''
    for key, value in vars(args).items():
        string += f'{key}: {value} |'
    lines.append(string)
    lines.append('\n')
    lines.append(str(weight_dict))
    # Add initial spaces to avoid markdown formatting in TensorBoard
    return '    ' + '\n    '.join(lines)

def train(args):
    # init training dataset
    print("preparing data...")
    train_dataset, train_dataloader = get_dataloader(args, "train", DC, augment=True)
    val_dataset, val_dataloader = get_dataloader(args, "val", DC, augment=False)
    dataloader = {
        "train": train_dataloader,
        "val": val_dataloader
    }

    print("initializing...")
    solver, num_params, root = get_solver(args, dataloader)
    print("Parameters: " + str(num_params/1000000)+" mil to train")

    print("Start training...\n")
    save_info(args, root, num_params, train_dataset, val_dataset)
    solver(args.epoch, args.verbose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, help="tag for the training, e.g. cuda_wl", default="")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--batch_size", type=int, help="batch size", default=7)
    parser.add_argument("--epoch", type=int, help="number of epochs", default=50)
    parser.add_argument("--verbose", type=int, help="iterations of showing verbose", default=10)
    parser.add_argument("--val_step", type=int, help="iterations of validating", default=1000)
    parser.add_argument("--lr", type=float, help="learning rate", default=2e-4) # default 1e-3
    parser.add_argument("--wd", type=float, help="weight decay", default=1e-6) # default 1e-6
    parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
    parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_augment", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_lang_cls", action="store_true", help="Do NOT use language classifier.")
    parser.add_argument("--no_detection", action="store_true", help="Do NOT train the detection module.")
    parser.add_argument("--no_reference", action="store_true", help="Do NOT train the localization module.")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
    parser.add_argument("--use_bidir", action="store_true", help="Use bi-directional GRU.")
    parser.add_argument("--use_pretrained", type=str, help="Specify the folder name containing the pretrained detection module.")
    parser.add_argument("--use_checkpoint", type=str, help="Specify the checkpoint root", default="")
    #chunking
    parser.add_argument("--use_chunking", action="store_true", help="Chunking")
    parser.add_argument("--max_chunk_size", type=int, help="max number of descriptions in one chunk", default=8)
    #language module
    parser.add_argument("--lang_module", type=str, default='gru', help="Language modules: gru, bert")
    parser.add_argument("--lr_bert", type=float, help="learning rate for bert", default=5e-5)
    parser.add_argument("--bert_wd", type=float, help="weight decay for Language module", default=1e-6)
    parser.add_argument("--num_bert_layers", type=int, help="bert layers", default=3)
    #match module
    parser.add_argument("--match_module", type=str, default='scanrefer', help="Match modules: scanrefer, dvg, transformer")
    parser.add_argument("--use_dist_weight_matrix", action="store_true", help="For the dvg matching module, should improve performance")
    parser.add_argument("--dvg_plus", action="store_true", help="Regularization for the training")
    parser.add_argument("--m_enc_layers", type=int, default=5, help="Amount of encoder layers for matching module when using vanilla transformer")
    parser.add_argument("--vt_dim_feed", type=int, default=1080, help="dimension of feedforward layers for matching modules encoder when using vanilla transformer")
    parser.add_argument("--vt_drop", type=float, default=0.1, help="Dropout for matching module when using vanilla transformer")
    parser.add_argument("--lr_match", default=5e-5, type=float)
    parser.add_argument("--match_wd", type=float, help="weight decay for Language module", default=1e-6)
    # detection module
    parser.add_argument("--detection_module", type=str, default='votenet', help="Detection modules: votenet, 3detr")
    parser.add_argument("--int_layers", action="store_true", help="Use the intermediate layers of 3DETR for the ref loss")
    # 3DETR optimizer
    parser.add_argument("--detr_lr", default=5e-4, type=float)
    parser.add_argument("--warm_lr", default=1e-6, type=float)
    parser.add_argument("--warm_lr_epochs", default=9, type=int)
    parser.add_argument("--final_lr", default=1e-6, type=float)
    parser.add_argument("--lr_scheduler", default="cosine", type=str)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--filter_biases_wd", default=False, action="store_true")
    parser.add_argument(
        "--clip_gradient", default=0.1, type=float, help="Max L2 norm of the gradient"
    )
    parser.add_argument("--sep_optim", action="store_true", help="Use seperate optimizers during training")
    parser.add_argument("--copy_paste", action="store_true", help="Use copy paste")
    args = parser.parse_args()

        
    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    train(args)
    
