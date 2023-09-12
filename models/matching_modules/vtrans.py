import torch.nn as nn
from argparse import Namespace

from .utils import (
    copy_paste,
    expand_object_features_mask, 
    fuse_objmask_match, 
    get_objectness_masks
)

class VTransMatchModule(nn.Module):
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
        assert self.args.detection_module in ["3detr", "votenet"], "detection_module must be either 3detr or votenet"
        
        # single vanilla Transformer encoder layer        
        _encoder_layer = nn.TransformerEncoderLayer(
                d_model=lang_size+128,
                nhead=8,
                dim_feedforward=args.vt_dim_feed,
                dropout=args.vt_drop,
            )
        # multi-layer Transformer encoder
        self.vt_fuse = nn.TransformerEncoder(
            encoder_layer=_encoder_layer, 
            num_layers=args.m_enc_layers
        )
        # reduce spatial dimension
        self.reduce = nn.Sequential(
            nn.Conv1d(lang_size + 128
                      , hidden_size, 1),
            nn.ReLU(),
        )
        
        # matching module
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
        Forward pass of the vanilla transformer matching module.

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

        # unpack outputs from language branch
        lang_feat = data_dict["lang_emb"] # batch_size * len_nun_max, lang_size
        lang_feat = lang_feat.unsqueeze(1).repeat(1, self.num_proposals, 1) # batch_size * len_nun_max, num_proposals, lang_size
        
        if self.args.use_chunking:
            batchsize, chunk_size = data_dict['ref_center_label_list'].shape[:2]
        else:
            batchsize = data_dict['ref_center_label'].shape[0]

        # train with copy-paste augmentation
        if self.args.copy_paste:
            features = copy_paste(
                data_dict=data_dict,
                features=features,
                objectness_masks=objectness_masks,
                batchsize=batchsize,
                num_proposals=self.num_proposals,
            )

        if self.args.use_chunking:
            # expand the features and objectness masks
            features, objectness_masks = expand_object_features_mask(features, objectness_masks, chunk_size)

        # fuse and match
        data_dict["cluster_ref"] = fuse_objmask_match(
            fusion_network=self.vt_fuse,
            matching_network=self.match,
            features=features,
            lang_feat=lang_feat,
            objectness_masks=objectness_masks
        )

        return data_dict
