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


def create_chunked_data(data: list, max_chunk_size: int):
    """
    Create training data chunks for objects in the same scene.
    The maximum number of objects in a chunk is max_chunk_size. 
    Chunks can be smaller if there are less objects in a scene.
    """
    data_chunked = []
    new_scene = []
    scene_id = ""
    for d in data:
        if scene_id != d["scene_id"]:
            scene_id = d["scene_id"]
            if len(new_scene) > 0:
                data_chunked.append(new_scene)
            new_scene = []
        if len(new_scene) >= max_chunk_size:
            data_chunked.append(new_scene)
            new_scene = []
        new_scene.append(d)
    
    data_chunked.append(new_scene)
    return data_chunked

def get_scanrefer(args, num_scenes, max_chunk_size):
    scanrefer_train = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
    scanrefer_val = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))        
    
    # get initial scene list
    train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
    val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_val])))
    if num_scenes == -1: 
        num_scenes = len(train_scene_list)
    else:
        assert len(train_scene_list) >= num_scenes
    
    # slice train_scene_list
    train_scene_list = train_scene_list[:num_scenes]

    # filter data in chosen scenes
    filtered_scanrefer_train = []
    for data in scanrefer_train:
        if data["scene_id"] in train_scene_list:
            filtered_scanrefer_train.append(data)

    if args.use_chunking:            
        scanrefer_train_chunked = create_chunked_data(filtered_scanrefer_train, max_chunk_size)
        scanrefer_val_chunked = create_chunked_data(scanrefer_val, max_chunk_size)

        train_chunked_lengths = [len(chunk) for chunk in scanrefer_train_chunked]
        val_chunked_lengths = [len(chunk) for chunk in scanrefer_val_chunked]

        assert sum(train_chunked_lengths) == len(filtered_scanrefer_train), "Chunking error, train"
        assert sum(val_chunked_lengths) == len(scanrefer_val), "Chunking error, val"
        print("Size of chunked dataset", len(scanrefer_train_chunked), len(scanrefer_val_chunked), len(scanrefer_train_chunked[0]))  # 4819 1253 8
    else:
        scanrefer_train_chunked = None
        scanrefer_val_chunked = None

    # all scanrefer scene
    all_scene_list = train_scene_list + val_scene_list
    print(f"train on {len(filtered_scanrefer_train)} samples and val on {len(scanrefer_val)} samples")
    return filtered_scanrefer_train, scanrefer_val, all_scene_list, scanrefer_train_chunked, scanrefer_val_chunked
    
def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])
    return scene_list

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