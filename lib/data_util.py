from copy import deepcopy
import json
import os
from torch.utils.data import DataLoader
import sys
sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.dataset import ScannetReferenceDataset
from lib.config import CONF

# constants
DC = ScannetDatasetConfig()
SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))

def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])
    return scene_list

def get_scanrefer(args, num_scenes, max_chunk_size):
    scanrefer_train = SCANREFER_TRAIN
    scanrefer_val = SCANREFER_VAL

    if args.no_reference:
        train_scene_list = get_scannet_scene_list("train")
        new_scanrefer_train = []
        for scene_id in train_scene_list:
            data = deepcopy(SCANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            new_scanrefer_train.append(data)

        val_scene_list = get_scannet_scene_list("val")
        new_scanrefer_val = []
        for scene_id in val_scene_list:
            data = deepcopy(SCANREFER_VAL[0])
            data["scene_id"] = scene_id
            new_scanrefer_val.append(data)
        
    else:
        # get initial scene list
        train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
        val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_val])))
        if num_scenes == -1: 
            num_scenes = len(train_scene_list)
        else:
            assert len(train_scene_list) >= num_scenes
        
        # slice train_scene_list
        train_scene_list = train_scene_list[:num_scenes]

        if args.use_chunking:
            # filter data in chosen scenes
            new_scanrefer_train = []
            scanrefer_train_chunked = []
            scanrefer_train_new_scene = []
            scene_id = ""
            for data in scanrefer_train:
                if data["scene_id"] in train_scene_list:
                    new_scanrefer_train.append(data)
                    if scene_id != data["scene_id"]:
                        scene_id = data["scene_id"]
                        if len(scanrefer_train_new_scene) > 0:
                            scanrefer_train_chunked.append(scanrefer_train_new_scene)
                        scanrefer_train_new_scene = []
                    if len(scanrefer_train_new_scene) >= max_chunk_size:
                        scanrefer_train_chunked.append(scanrefer_train_new_scene)
                        scanrefer_train_new_scene = []
                    scanrefer_train_new_scene.append(data)
                    
            scanrefer_train_chunked.append(scanrefer_train_new_scene)

            new_scanrefer_val = scanrefer_val
            scanrefer_val_chunked = []
            scanrefer_val_new_scene = []
            scene_id = ""
            for data in scanrefer_val:
                if scene_id != data["scene_id"]:
                    scene_id = data["scene_id"]
                    if len(scanrefer_val_new_scene) > 0:
                        scanrefer_val_chunked.append(scanrefer_val_new_scene)
                    scanrefer_val_new_scene = []
                if len(scanrefer_val_new_scene) >= max_chunk_size:
                    scanrefer_val_chunked.append(scanrefer_val_new_scene)
                    scanrefer_val_new_scene = []
                scanrefer_val_new_scene.append(data)
            scanrefer_val_chunked.append(scanrefer_val_new_scene)
        else:
        # filter data in chosen scenes
            new_scanrefer_train = []
            for data in scanrefer_train:
                if data["scene_id"] in train_scene_list:
                    new_scanrefer_train.append(data)

            new_scanrefer_val = scanrefer_val

    if args.use_chunking:
        print("scanrefer_train_new", len(scanrefer_train_chunked), len(scanrefer_val_chunked), len(scanrefer_train_chunked[0]))  # 4819 1253 8
        sum = 0
        for i in range(len(scanrefer_train_chunked)):
            sum += len(scanrefer_train_chunked[i])
        print("training sample numbers", sum)  # 36665
        # all scanrefer scene
        all_scene_list = train_scene_list + val_scene_list
        print("train on {} samples and val on {} samples".format(len(new_scanrefer_train), len(new_scanrefer_val)))  # 36665 9508
        return new_scanrefer_train, new_scanrefer_val, all_scene_list, scanrefer_train_chunked, scanrefer_val_chunked
    else:
        # all scanrefer scene
        all_scene_list = train_scene_list + val_scene_list
        print("train on {} samples and val on {} samples".format(len(new_scanrefer_train), len(new_scanrefer_val)))
        return new_scanrefer_train, new_scanrefer_val, all_scene_list
    

def get_dataloader(args, split, config, augment):
    dataset = ScannetReferenceDataset(
        num_scenes=args.num_scenes,
        split=split, 
        num_points=args.num_points, 
        use_height=(not args.no_height),
        use_color=args.use_color, 
        use_normal=args.use_normal, 
        use_multiview=args.use_multiview,
        augment=augment,
        #chunking
        chunking = args.use_chunking,
        max_chunk_size=args.max_chunk_size,
        # language module
        lang_module = args.lang_module
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return dataset, dataloader