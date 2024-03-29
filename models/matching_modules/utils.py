import torch
import torch.nn as nn

def expand_object_features_mask(
        features: torch.Tensor, 
        objectness_masks: torch.Tensor, 
        chunk_size: int, 
    ) -> tuple:
    """
    Expand the features and objectness masks to match the chunked language features.

    Args:
    - features (tensor): Object features.
    - objectness_masks (tensor): Objectness masks.
    - chunk_size (int): Language chunk size.

    Returns:
    - tuple: Expanded features and objectness masks.
    """
    features = features.unsqueeze(1).repeat(1, chunk_size, 1, 1)
    batchsize, _, d3, d4 = features.shape[:4]
    features = features.reshape(batchsize * chunk_size, d3, d4)
    objectness_masks = objectness_masks.unsqueeze(1)\
        .repeat(1, chunk_size, 1, 1)\
        .reshape(batchsize * chunk_size, d3, 1)
    return features, objectness_masks


def fuse_objmask_match(
        fusion_network: nn.Module,
        matching_network: nn.Module,
        features: torch.Tensor, 
        lang_feat: torch.Tensor,
        objectness_masks: torch.Tensor
    ) -> torch.Tensor:
    """
    Concat object features and language features and then combine the information with the fusion network.
    The size of the fused features is (batch_size, num_proposals, object_feature_size + lang_size).
    Apply objectness masks to the fused features to mask out invalid proposals.
    Finally, compute confidence scores the a box belongs to a description with the matching network.

    Args:
    - fusion_network (nn.Module): The fusion network.
    - matching_network (nn.Module): The matching network.
    - features (tensor): Object features.
    - lang_feat (tensor): Language features.
    - objectness_masks (tensor): Objectness masks.

    Returns:
    - tensor: Confidence scores.
    """
    # fuse
    features = torch.cat([features, lang_feat], dim=-1) # batch_size, num_proposals, 128 + lang_size
    features = features.permute(0, 2, 1).contiguous() # batch_size, 128 + lang_size, num_proposals
    
    # fusion network
    features = fusion_network(features) # batch_size, hidden_size, num_proposals
    # mask out invalid proposals
    objectness_masks = objectness_masks.permute(0, 2, 1).contiguous() # batch_size, 1, num_proposals
    features = features * objectness_masks

    # match
    confidences = matching_network(features).squeeze(1) # batch_size, num_proposals 
    return confidences


def get_objectness_masks(data_dict: dict, detection_module: str):
    """
    Retrieve objectness masks from the detection branch.

    Args:
    - data_dict (dict): A dictionary containing:
        - outputs (dict): if detection_module is 3detr
        - objectness_scores (tensor): if detection_module is votenet
    - detection_module (str): The detection module used. Either 3detr or votenet.

    Returns:
    - tensor: Objectness masks.
    """
    assert detection_module in ["3detr", "votenet"], "detection_module must be either 3detr or votenet"
    if detection_module == "3detr":
        objectness_masks = torch.as_tensor(
            (data_dict['outputs']["objectness_prob"].unsqueeze(-1))>0.5,
            dtype=torch.float32
        )
    elif detection_module == "votenet":
        objectness_masks = data_dict['objectness_scores'].max(2)[1].float().unsqueeze(2) # batch_size, num_proposals, 1
    
    return objectness_masks


def copy_paste(
        data_dict: dict,
        features: torch.Tensor,
        objectness_masks: torch.Tensor,
        batchsize: int,
        num_proposals: int,    
    ):
    """
    Copy paste method to increase training difficulty.
    https://github.com/zlccccc/3DVG-Transformer/blob/main/models/match_module.py

    Args:
    - data_dict (dict): A dictionary containing:
        - istrain (tensor): A tensor indicating whether the current batch is for training.
        - random (tensor): A tensor indicating whether to apply copy paste.
    - features (tensor): Object features.
    - objectness_masks (tensor): Objectness masks.
    - batchsize (int): Batch size.
    - num_proposals (int): Number of box proposals.

    Returns:
    - tensor: Modified object features.
    """

    feature_modified = features.clone()
    # This is some random application of objectness mask
    if data_dict["istrain"][0] == 1 and data_dict["random"] < 0.5:
        obj_masks = objectness_masks.bool().squeeze(2)  # batch_size, num_proposals
        obj_lens = torch.zeros(batchsize, dtype=torch.int).cuda()
        for i in range(batchsize):
            obj_mask = torch.where(obj_masks[i, :] == True)[0]
            obj_len = obj_mask.shape[0]
            obj_lens[i] = obj_len

        obj_masks_reshape = obj_masks.reshape(batchsize*num_proposals)
        obj_features = features.reshape(batchsize*num_proposals, -1)
        obj_mask = torch.where(obj_masks_reshape[:] == True)[0]
        total_len = obj_mask.shape[0]
        obj_features = obj_features[obj_mask, :].repeat(2,1)  # total_len, hidden_size
        j = 0
        for i in range(batchsize):
            obj_mask = torch.where(obj_masks[i, :] == False)[0]
            obj_len = obj_mask.shape[0]
            j += obj_lens[i]
            if obj_len < total_len - obj_lens[i]:
                feature_modified[i, obj_mask, :] = obj_features[j:j + obj_len, :]
            else:
                feature_modified[i, obj_mask[:total_len - obj_lens[i]], :] = obj_features[j:j + total_len - obj_lens[i], :]

    return feature_modified