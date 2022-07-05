import torch
import torch.nn as nn
import random

class MatchModule(nn.Module):
    def __init__(self, args, num_proposals=256, lang_size=256, hidden_size=128):
        super().__init__() 

        self.args = args
        self.num_proposals = num_proposals
        self.lang_size = lang_size
        self.hidden_size = hidden_size
        
        self.fuse = nn.Sequential(
            nn.Conv1d(self.lang_size + 128, hidden_size, 1),
            nn.ReLU()
        )
        # self.match = nn.Conv1d(hidden_size, 1, 1)
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
            if self.args.int_layers:
                int_objectness_masks = torch.zeros((len(data_dict['aux_outputs']),data_dict['aux_outputs'][0]['objectness_prob'].shape[0], data_dict['aux_outputs'][0]['objectness_prob'].shape[1], 1)).cuda()
                int_features = torch.zeros((len(data_dict['aux_outputs']),data_dict['aux_outputs'][0]['box_features'].shape[0], data_dict['aux_outputs'][0]['box_features'].shape[1], data_dict['aux_outputs'][0]['box_features'].shape[2])).cuda()
    
                for l in range(len(data_dict['aux_outputs'])):
                    int_objectness_masks[l] = torch.as_tensor((data_dict['aux_outputs'][l]['objectness_prob'].unsqueeze(-1))>0.5,dtype=torch.float32)
                    int_features[l] = data_dict['aux_outputs'][l]['box_features']
        elif self.args.detection_module == "votenet":
            objectness_masks = data_dict['objectness_scores'].max(2)[1].float().unsqueeze(2) # batch_size, num_proposals, 1

        # unpack outputs from language branch
        lang_feat = data_dict["lang_emb"] # batch_size * len_nun_max, lang_size
        
        #for key in data_dict:
        #    print(f'{key}: {data_dict[key]}')

        if self.args.use_chunking:
            data_dict["random"] = random.random()
            batchsize, len_nun_max = data_dict['ref_center_label_list'].shape[:2]
            # print(f'batchsize, len_nun_max: {batchsize}, {len_nun_max}')
            features = features.unsqueeze(1).repeat(1, len_nun_max, 1, 1)
            v1, v2, v3, v4 = features.shape[:4]
            #print(f'v1: {v1}, v2: {v2}, v3: {v3}, v4: {v4}') # batchsize, len_nun
            features = features.reshape(batchsize * len_nun_max, v3, v4)
            #print(f'feature shape after unsquezze: {features.shape}')
            #print(f'objectness_masks shape before unsquezze: {objectness_masks.shape}')
            objectness_masks = objectness_masks.unsqueeze(1).repeat(1, len_nun_max, 1, 1).reshape(batchsize * len_nun_max, v3, 1)
            #print(f'objectness_masks shape after unsquezze: {objectness_masks.shape}')
            if self.args.detection_module == "3detr" and self.args.int_layers:
                int_features_2 = torch.zeros((int_features.shape[0], int_features.shape[1], len_nun_max, int_features.shape[2], int_features.shape[3])).cuda()
                int_features_3 = torch.zeros((int_features.shape[0], int_features.shape[1]*len_nun_max, int_features.shape[2], int_features.shape[3])).cuda()
                int_objectness_masks_2 = torch.zeros((int_objectness_masks.shape[0], int_objectness_masks.shape[1]*len_nun_max, int_objectness_masks.shape[2], int_objectness_masks.shape[3])).cuda()
                for l in range(int_features.shape[0]):
                    int_features_2[l] = int_features[l].unsqueeze(1).repeat(1, len_nun_max, 1, 1)
                    v1, v2, v3, v4 = int_features_2[l].shape[:4]
                    int_features_3[l] = int_features_2[l].reshape(batchsize * len_nun_max, v3, v4)
                    int_objectness_masks_2[l] = int_objectness_masks[l].unsqueeze(1).repeat(1, len_nun_max, 1, 1).reshape(batchsize * len_nun_max, v3, 1)
        else:
            pass

        #print(f'lang_feat shape: {lang_feat.shape}')
        lang_feat = lang_feat.unsqueeze(1).repeat(1, self.num_proposals, 1) # batch_size, num_proposals, lang_size
        #print(f'lang_feat shape after unsquezze: {lang_feat.shape}')

        # fuse
        features = torch.cat([features, lang_feat], dim=-1) # batch_size, num_proposals, 128 + lang_size
        features = features.permute(0, 2, 1).contiguous() # batch_size, 128 + lang_size, num_proposals
        # fuse features
        features = self.fuse(features) # batch_size, hidden_size, num_proposals
        # mask out invalid proposals
        objectness_masks = objectness_masks.permute(0, 2, 1).contiguous() # batch_size, 1, num_proposals
        features = features * objectness_masks

        # match
        confidences = self.match(features).squeeze(1) # batch_size, num_proposals
                
        data_dict["cluster_ref"] = confidences
        
        if self.args.detection_module == "3detr" and self.args.int_layers:
            int_features_cat = torch.zeros((int_features_3.shape[0], int_features_3.shape[1], int_features_3.shape[2], int_features_3.shape[3]+lang_feat.shape[2])).cuda()
            for l in range(len(int_features_3)):
                
                int_features_cat[l] = torch.cat([int_features_3[l], lang_feat], dim=-1)
                int_features_cat_layer = int_features_cat[l].permute(0, 2, 1).contiguous().cuda()
                # fuse features
                int_features_cat_layer = self.fuse(int_features_cat_layer)
                # mask out invalid proposals
                int_objectness_masks_layer = int_objectness_masks_2[l].permute(0, 2, 1).contiguous()
                int_features_cat_layer = int_features_cat_layer * int_objectness_masks_layer
                
                confidences = self.match(int_features_cat_layer.squeeze(1))
                data_dict['aux_outputs'][l]['cluster_ref'] = confidences

        return data_dict
