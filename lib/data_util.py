from copy import deepcopy
import json
import os
from argparse import Namespace
from torch.utils.data import DataLoader
import sys
sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.dataset import ScannetReferenceDataset
from lib.config import CONF

# constants
DC = ScannetDatasetConfig()

def create_chunked_data(data: list, max_chunk_size: int) -> list:
    """
    Create training data chunks for objects in the same scene.
    The maximum number of objects in a chunk is max_chunk_size. 
    Chunks can be smaller if there are less objects in a scene.

    Args:
    - data (list): List of dictionaries containing the data.
    - max_chunk_size (int): Maximum number of objects in a chunk.

    Returns:
    - list: Data chunked into scences.
    """
    data_chunked = []
    new_scene = []
    scene_id = ""
    for d in data:
        if scene_id != d["scene_id"]:
            # when the scene changes, add the previous scene to the list
            scene_id = d["scene_id"]
            if len(new_scene) > 0:
                data_chunked.append(new_scene)
            new_scene = []
        if len(new_scene) >= max_chunk_size:
            # when the chunk is full, add it to the list
            data_chunked.append(new_scene)
            new_scene = []
        new_scene.append(d)
    
    data_chunked.append(new_scene)
    return data_chunked

def get_scanrefer(
        args: Namespace, 
        num_scenes: int, 
        max_chunk_size: int,
    ):
    """
    Get the ScanRefer data. If specified chunk the data based on objects in the same scene.

    Args:
    - args (Namespace): Arguments.
    - num_scenes (int): Number of scenes to use. -1 all scence.
    - max_chunk_size (int): Maximum number of objects in a chunk.

    Returns:
    - dict: ScanRefer data split into train val.
    - list: All objects in the ScanRefer data validation and training.
    - dict: ScanRefer data chunked into scenes. Split into train val. None if chunking is not used.
    """

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

    scanrefer = {
        "train": scanrefer_train,
        "val": scanrefer_val
    }
    scanrefer_chunked = {
        "train": scanrefer_train_chunked,
        "val": scanrefer_val_chunked
    }

    return scanrefer, all_scene_list, scanrefer_chunked
    
def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])
    return scene_list

def get_dataloader(
        args: Namespace, 
        split: str, 
        config: ScannetDatasetConfig, 
        augment: bool = False
    ):
    """
    Create a dataloader for the ScanRefer dataset.
    """

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
        chunk_size=args.max_chunk_size,
        # language module
        lang_module = args.lang_module
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return dataset, dataloader