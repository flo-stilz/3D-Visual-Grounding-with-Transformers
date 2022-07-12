import torch
import torch.nn as nn
import random
import numpy as np
from models.transformer.attention import MultiHeadAttention
from utils.box_util import box3d_iou_batch

class VTransMatchModule(nn.Module):
    def __init__(self, args, num_proposals=256, lang_size=256, hidden_size=128):
        super().__init__() 

        self.args = args
        self.num_proposals = num_proposals
        self.lang_size = lang_size
        self.hidden_size = hidden_size
        self.num_encoder_layers = args.m_enc_layers
        self.dim_feedforward = args.vt_dim_feed
        self.vt_drop = args.vt_drop
        
        # vanilla Transformer encoder layer        
        self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.lang_size+128,
                nhead=8,
                dim_feedforward=self.dim_feedforward,
                dropout=self.vt_drop,
            )
        self.vt_fuse = nn.TransformerEncoder(self.encoder_layer, self.num_encoder_layers)
        self.reduce = nn.Sequential(
            nn.Conv1d(self.lang_size + 128
                      , hidden_size, 1),
            nn.ReLU(),
        )
        
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
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """

        # unpack outputs from detection branch
        features = data_dict['aggregated_vote_features'] # batch_size, num_proposal, 128
        if self.args.detection_module == "3detr":
            objectness_masks = torch.as_tensor((data_dict['outputs']["objectness_prob"].unsqueeze(-1))>0.5,dtype=torch.float32)
        elif self.args.detection_module == "votenet":
            objectness_masks = data_dict['objectness_scores'].max(2)[1].float().unsqueeze(2) # batch_size, num_proposals, 1

        # unpack outputs from language branch
        lang_feat = data_dict["lang_emb"] # batch_size * len_nun_max, lang_size
        lang_feat = lang_feat.unsqueeze(1).repeat(1, self.num_proposals, 1) # batch_size * len_nun_max, num_proposals, lang_size
        

        if self.args.use_chunking:
            batchsize, len_nun_max = data_dict['ref_center_label_list'].shape[:2]
        else:
            batchsize = data_dict['ref_center_label'].shape[0]

        
        #---------------copy paste----------------------
        data_dict["random"] = random.random()
        # copy paste part
        
        if self.args.copy_paste:
            feature0 = features.clone()
            # This is some random application of objectness mask
            if data_dict["istrain"][0] == 1 and data_dict["random"] < 0.5:
                obj_masks = objectness_masks.bool().squeeze(2)  # batch_size, num_proposals
                obj_lens = torch.zeros(batchsize, dtype=torch.int).cuda()
                for i in range(batchsize):
                    obj_mask = torch.where(obj_masks[i, :] == True)[0]
                    obj_len = obj_mask.shape[0]
                    obj_lens[i] = obj_len


                obj_masks_reshape = obj_masks.reshape(batchsize*self.num_proposals)
                obj_features = features.reshape(batchsize*self.num_proposals, -1)
                obj_mask = torch.where(obj_masks_reshape[:] == True)[0]
                total_len = obj_mask.shape[0]
                obj_features = obj_features[obj_mask, :].repeat(2,1)  # total_len, hidden_size
                j = 0
                for i in range(batchsize):
                    obj_mask = torch.where(obj_masks[i, :] == False)[0]
                    obj_len = obj_mask.shape[0]
                    j += obj_lens[i]
                    if obj_len < total_len - obj_lens[i]:
                        feature0[i, obj_mask, :] = obj_features[j:j + obj_len, :]
                    else:
                        feature0[i, obj_mask[:total_len - obj_lens[i]], :] = obj_features[j:j + total_len - obj_lens[i], :]

            if self.args.use_chunking:
                feature0 = feature0.unsqueeze(1).repeat(1, len_nun_max, 1, 1)
                v1, v2, v3, v4 = feature0.shape[:4]
                feature0 = feature0.reshape(batchsize * len_nun_max, v3, v4)
                objectness_masks = objectness_masks.unsqueeze(1).repeat(1, len_nun_max, 1, 1).reshape(batchsize * len_nun_max, v3, 1)

            features1 = torch.cat([feature0, lang_feat], dim=-1)
        #---------------copy paste end-------------------
        else:
            if self.args.use_chunking:
                features = features.unsqueeze(1).repeat(1, len_nun_max, 1, 1)
                v1, v2, v3, v4 = features.shape[:4]
                features = features.reshape(batchsize * len_nun_max, v3, v4)
                objectness_masks = objectness_masks.unsqueeze(1).repeat(1, len_nun_max, 1, 1).reshape(batchsize * len_nun_max, v3, 1)
            features1 = torch.cat([features, lang_feat], dim=-1)
        
        # fuse
        features1 = features1.permute(1, 0, 2).contiguous() # num_proposals, batch_size, 128 + lang_size
        
        # Apply self-attention on fused features
        features1 = self.vt_fuse(features1) # num_proposals, batch_size, lang_size + 128
        features1 = features1.permute(1, 2, 0).contiguous() # batch_size, lang_size +128, num_proposals
    
        features1 = self.reduce(features1) # batch_size, hidden_size, num_proposals
        # mask out invalid proposals
        objectness_masks = objectness_masks.permute(0, 2, 1).contiguous() # batch_size, 1, num_proposals
        features1 = features1 * objectness_masks

        # match
        confidences = self.match(features1).squeeze(1) # batch_size, num_proposals
                
        data_dict["cluster_ref"] = confidences

        return data_dict
