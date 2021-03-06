import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer.attention import MultiHeadAttention
from models.transformer.utils import PositionWiseFeedForward
import random


class DVGMatchModule(nn.Module):
    def __init__(self, args, num_proposals=256, lang_size=256, hidden_size=128, lang_num_size=300, det_channel=288*4, head=4, depth=2):
        super().__init__()
        self.use_dist_weight_matrix = True  ## False: initial 3DVG-Transformer

        self.num_proposals = num_proposals
        self.lang_size = lang_size
        self.hidden_size = hidden_size
        self.depth = depth - 1
        self.args = args

        self.features_concat = nn.Sequential(
            nn.Conv1d(det_channel, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, 1),
        )
        self.match = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Conv1d(hidden_size, 1, 1)
        )
        self.self_attn = nn.ModuleList(
            MultiHeadAttention(d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head) for i in range(depth))
        self.cross_attn = nn.ModuleList(
            MultiHeadAttention(d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head) for i in range(depth))  # k, q, v
        # d_model = 128, d_k = 32, d_v = 32, h = 4

        self.bbox_embedding = nn.Linear(12, 128)

    def forward(self, data_dict):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """
        # --------- 3DVG + part ------------
        if self.use_dist_weight_matrix:
            if self.args.detection_module == "3detr":
                objects_center = data_dict['outputs']["center_normalized"]
            elif self.args.detection_module == "votenet":
                objects_center = data_dict['center']

            # Attention Weight
            N_K = objects_center.shape[1]
            center_A = objects_center[:, None, :, :].repeat(1, N_K, 1, 1)
            center_B = objects_center[:, :, None, :].repeat(1, 1, N_K, 1)
            dist = (center_A - center_B).pow(2)
            # print(dist.shape, '<< dist shape', flush=True)
            dist = torch.sqrt(torch.sum(dist, dim=-1))[:, None, :, :]
            dist_weights = 1 / (dist+1e-2)
            norm = torch.sum(dist_weights, dim=2, keepdim=True)
            dist_weights = dist_weights / norm
            zeros = torch.zeros_like(dist_weights)

            # slightly different with our ICCV paper, which leads to higher results (3DVG-Transformer+)
            dist_weights = torch.cat([dist_weights, -dist, zeros, zeros], dim=1).detach()
            attention_matrix_way = 'add'
        else:
            dist_weights = None
            attention_matrix_way = 'mul'
        # --------- End ------------

        # select detection module
        if self.args.detection_module == '3detr':
            features = data_dict['aggregated_vote_features']
            objectness_masks = torch.as_tensor((data_dict['outputs']["objectness_prob"].unsqueeze(-1))>0.5,dtype=torch.float32)
        elif self.args.detection_module == 'votenet':
            features = data_dict['aggregated_vote_features'] # batch_size, num_proposal, 128
            objectness_masks = data_dict['objectness_scores'].max(2)[1].float().unsqueeze(2)  # batch_size, num_proposals, 1
        # end
        
        batch_size, num_proposal = features.shape[:2]
        data_dict["random"] = random.random()

        # self attention part on the feature level
        features = self.self_attn[0](features, features, features, attention_weights=dist_weights, way=attention_matrix_way)

        
        if self.args.use_chunking:
            batch_size, len_nun_max = data_dict['ref_center_label_list'].shape[:2]

        # copy paste part
        feature0 = features.clone()
        if self.args.copy_paste:
            # This is some random application of objectness mask
            if data_dict["istrain"][0] == 1 and data_dict["random"] < 0.5:
                obj_masks = objectness_masks.bool().squeeze(2)  # batch_size, num_proposals
                obj_lens = torch.zeros(batch_size, dtype=torch.int).cuda()
                for i in range(batch_size):
                    obj_mask = torch.where(obj_masks[i, :] == True)[0]
                    obj_len = obj_mask.shape[0]
                    obj_lens[i] = obj_len

                obj_masks_reshape = obj_masks.reshape(batch_size*num_proposal)
                obj_features = features.reshape(batch_size*num_proposal, -1)
                obj_mask = torch.where(obj_masks_reshape[:] == True)[0]
                total_len = obj_mask.shape[0]
                obj_features = obj_features[obj_mask, :].repeat(2,1)  # total_len, hidden_size
                j = 0
                for i in range(batch_size):
                    obj_mask = torch.where(obj_masks[i, :] == False)[0]
                    obj_len = obj_mask.shape[0]
                    j += obj_lens[i]
                    if obj_len < total_len - obj_lens[i]:
                        feature0[i, obj_mask, :] = obj_features[j:j + obj_len, :]
                    else:
                        feature0[i, obj_mask[:total_len - obj_lens[i]], :] = obj_features[j:j + total_len - obj_lens[i], :]
        
        if self.args.lang_module == 'gru':
            lang_fea = data_dict["lang_fea"]
        elif self.args.lang_module == 'bert':
            lang_fea = data_dict["lang_emb"].unsqueeze(1)   

        feature1 = feature0[:, None, :, :].repeat(1, len_nun_max, 1, 1).reshape(batch_size*len_nun_max, num_proposal, -1)
        if dist_weights is not None:
            dist_weights = dist_weights[:, None, :, :, :].repeat(1, len_nun_max, 1, 1, 1).reshape(batch_size*len_nun_max, dist_weights.shape[1], num_proposal, num_proposal)

        feature1 = self.cross_attn[0](feature1, lang_fea, lang_fea, data_dict["attention_mask"]) # query, key, value, 

        for _ in range(self.depth):
            feature1 = self.self_attn[_+1](feature1, feature1, feature1, attention_weights=dist_weights, way=attention_matrix_way)
            feature1 = self.cross_attn[_+1](feature1, lang_fea, lang_fea, data_dict["attention_mask"])

        # match
        feature1_agg = feature1
        feature1_agg = feature1_agg.permute(0, 2, 1).contiguous()

        confidence = self.match(feature1_agg).squeeze(1)  # batch_size, num_proposals
        data_dict["cluster_ref"] = confidence

        return data_dict


