import torch
import torch.nn as nn
from argparse import Namespace

from .utils import (
    expand_object_features_mask, 
    fuse_objmask_match, 
    get_objectness_masks
)

class MatchModule(nn.Module):
    def __init__(
            self, 
            args: Namespace, 
            num_proposals: int = 256, 
            lang_size: int = 256, 
            hidden_size: int = 128
        ) -> None:
        """
        Args:
        - args: config file
        - num_proposals: number of proposals
        - lang_size: size of language embeddings
        - hidden_size: size of hidden layer
        """
        super().__init__() 

        self.args = args
        self.num_proposals = num_proposals
        
        # MLP fusion network
        self.fuse = nn.Sequential(
            nn.Conv1d(lang_size + 128, hidden_size, 1),
            nn.ReLU()
        )

        # MLP matching network
        self.match = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, 1, 1)
        )

    def forward(self, data_dict):
        """
        Forward pass of the MLP matching module.

        Args:
        - data_dict (dict): A dictionary containing:
            - xyz: (B,K,3)
            - features: (B,C,K)
        Returns:
        - dict: Modified input dictionary containing the following keys:
            - scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """

        # unpack outputs from detection branch
        features = data_dict['aggregated_vote_features'] # batch_size, num_proposal, 128
        objectness_masks = get_objectness_masks(data_dict, self.args.detection_module)

        # expand the language features to match the number of proposals
        lang_feat = data_dict["lang_emb"] # batch_size * len_nun_max, lang_size
        lang_feat = lang_feat.unsqueeze(1).repeat(1, self.num_proposals, 1) # batch_size * len_nun_max, num_proposals, lang_size
        
        if self.args.use_chunking:
            # expand the features and objectness masks
            _, max_chunk_size = data_dict['ref_center_label_list'].shape[:2]
            features, objectness_masks = expand_object_features_mask(features, objectness_masks, max_chunk_size)

        # fuse and match
        data_dict["cluster_ref"] = fuse_objmask_match(
            fusion_network=self.fuse,
            matching_network=self.match,
            features=features,
            lang_feat=lang_feat,
            objectness_masks=objectness_masks
        )

        # use intermediate layer box features
        if self.args.detection_module == "3detr" and self.args.int_layers:
            int_objectness_masks = torch.zeros((len(data_dict['aux_outputs']),data_dict['aux_outputs'][0]['objectness_prob'].shape[0], data_dict['aux_outputs'][0]['objectness_prob'].shape[1], 1)).cuda()
            int_features = torch.zeros((len(data_dict['aux_outputs']),data_dict['aux_outputs'][0]['box_features'].shape[0], data_dict['aux_outputs'][0]['box_features'].shape[1], data_dict['aux_outputs'][0]['box_features'].shape[2])).cuda()
            for l in range(int_features):
                # intermediate layer objectness masks
                int_objectness_masks[l] = torch.as_tensor((data_dict['aux_outputs'][l]['objectness_prob'].unsqueeze(-1))>0.5,dtype=torch.float32)
                # intermediate layer box features
                int_features[l] = data_dict['aux_outputs'][l]['box_features']
                # expand the features and objectness masks
                int_features[l], int_objectness_masks[l] = expand_object_features_mask(int_features[l], int_objectness_masks[l], max_chunk_size)
                # fuse and match
                data_dict['aux_outputs'][l]['cluster_ref'] = fuse_objmask_match(
                    fusion_network=self.fuse,
                    matching_network=self.match,
                    features=int_features[l], 
                    lang_feat=lang_feat,
                    objectness_masks=int_objectness_masks[l]
                )
        
        return data_dict


    