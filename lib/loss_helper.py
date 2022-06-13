# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from utils.nn_distance import nn_distance, huber_loss
from lib.ap_helper import parse_predictions
from lib.loss import SoftmaxRankingLoss
from utils.box_util import get_3d_box, get_3d_box_batch, box3d_iou, box3d_iou_batch
import torch.nn.functional as F
from DETR.utils.box_util import generalized_box3d_iou
from scipy.optimize import linear_sum_assignment
from DETR.utils.dist import all_reduce_average

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3 # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8] # put larger weights on positive objectness

def compute_vote_loss(data_dict):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        data_dict: dict (read-only)
    
    Returns:
        vote_loss: scalar Tensor
            
    Overall idea:
        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to vote for the object center.

        Each seed point may vote for multiple translations v1,v2,v3
        A seed point may also be in the boxes of multiple objects:
        o1,o2,o3 with corresponding GT votes c1,c2,c3

        Then the loss for this seed point is:
            min(d(v_i,c_j)) for i=1,2,3 and j=1,2,3
    """

    # Load ground truth votes and assign them to seed points
    batch_size = data_dict['seed_xyz'].shape[0]
    num_seed = data_dict['seed_xyz'].shape[1] # B,num_seed,3
    vote_xyz = data_dict['vote_xyz'] # B,num_seed*vote_factor,3
    seed_inds = data_dict['seed_inds'].long() # B,num_seed in [0,num_points-1]

    # Get groundtruth votes for the seed points
    # vote_label_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
    seed_gt_votes_mask = torch.gather(data_dict['vote_label_mask'], 1, seed_inds)
    seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,3*GT_VOTE_FACTOR)
    seed_gt_votes = torch.gather(data_dict['vote_label'], 1, seed_inds_expand)
    seed_gt_votes += data_dict['seed_xyz'].repeat(1,1,3)

    # Compute the min of min of distance
    vote_xyz_reshape = vote_xyz.view(batch_size*num_seed, -1, 3) # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape = seed_gt_votes.view(batch_size*num_seed, GT_VOTE_FACTOR, 3) # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    votes_dist, _ = torch.min(dist2, dim=1) # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist = votes_dist.view(batch_size, num_seed)
    vote_loss = torch.sum(votes_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)
    return vote_loss

def compute_objectness_loss(data_dict):
    """ Compute objectness loss for the proposals.

    Args:
        data_dict: dict (read-only)

    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
        object_assignment: (batch_size, num_seed) Tensor with long int
            within [0,num_gt_object-1]
    """ 
    # Associate proposal and GT objects by point-to-point distances
    aggregated_vote_xyz = data_dict['aggregated_vote_xyz']
    gt_center = data_dict['center_label'][:,:,0:3]
    B = gt_center.shape[0]
    K = aggregated_vote_xyz.shape[1]
    K2 = gt_center.shape[1]
    dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center) # dist1: BxK, dist2: BxK2

    # Generate objectness label and mask
    # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
    # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
    euclidean_dist1 = torch.sqrt(dist1+1e-6)
    objectness_label = torch.zeros((B,K), dtype=torch.long).cuda()
    objectness_mask = torch.zeros((B,K)).cuda()
    objectness_label[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1>FAR_THRESHOLD] = 1

    # Compute objectness loss
    objectness_scores = data_dict['objectness_scores']
    criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
    objectness_loss = criterion(objectness_scores.transpose(2,1), objectness_label)
    objectness_loss = torch.sum(objectness_loss * objectness_mask)/(torch.sum(objectness_mask)+1e-6)

    # Set assignment
    object_assignment = ind1 # (B,K) with values in 0,1,...,K2-1

    return objectness_loss, objectness_label, objectness_mask, object_assignment



def compute_box_and_sem_cls_loss(data_dict, config):
    """ Compute 3D bounding box and semantic classification loss.

    Args:
        data_dict: dict (read-only)

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
        size_cls_loss
        size_reg_loss
        sem_cls_loss
    """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    object_assignment = data_dict['object_assignment']
    batch_size = object_assignment.shape[0]

    # Compute center loss
    pred_center = data_dict['center']
    gt_center = data_dict['center_label'][:,:,0:3]
    dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center) # dist1: BxK, dist2: BxK2
    box_label_mask = data_dict['box_label_mask']
    objectness_label = data_dict['objectness_label'].float()
    centroid_reg_loss1 = \
        torch.sum(dist1*objectness_label)/(torch.sum(objectness_label)+1e-6)
    centroid_reg_loss2 = \
        torch.sum(dist2*box_label_mask)/(torch.sum(box_label_mask)+1e-6)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2

    # Compute heading loss
    heading_class_label = torch.gather(data_dict['heading_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
    heading_class_loss = criterion_heading_class(data_dict['heading_scores'].transpose(2,1), heading_class_label) # (B,K)
    heading_class_loss = torch.sum(heading_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    heading_residual_label = torch.gather(data_dict['heading_residual_label'], 1, object_assignment) # select (B,K) from (B,K2)
    heading_residual_normalized_label = heading_residual_label / (np.pi/num_heading_bin)

    # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
    heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1], num_heading_bin).zero_()
    heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_heading_bin)
    heading_residual_normalized_loss = huber_loss(torch.sum(data_dict['heading_residuals_normalized']*heading_label_one_hot, -1) - heading_residual_normalized_label, delta=1.0) # (B,K)
    heading_residual_normalized_loss = torch.sum(heading_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # Compute size loss
    size_class_label = torch.gather(data_dict['size_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    
    criterion_size_class = nn.CrossEntropyLoss(reduction='none')
    size_class_loss = criterion_size_class(data_dict['size_scores'].transpose(2,1), size_class_label) # (B,K)
    size_class_loss = torch.sum(size_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)
    
    size_residual_label = torch.gather(data_dict['size_residual_label'], 1, object_assignment.unsqueeze(-1).repeat(1,1,3)) # select (B,K,3) from (B,K2,3)
    size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
    size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_size_cluster)
    size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1,1,1,3) # (B,K,num_size_cluster,3)
    predicted_size_residual_normalized = torch.sum(data_dict['size_residuals_normalized']*size_label_one_hot_tiled, 2) # (B,K,3)

    mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0) # (1,1,num_size_cluster,3) 
    mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2) # (B,K,3)
    size_residual_label_normalized = size_residual_label / mean_size_label # (B,K,3)
    size_residual_normalized_loss = torch.mean(huber_loss(predicted_size_residual_normalized - size_residual_label_normalized, delta=1.0), -1) # (B,K,3) -> (B,K)
    size_residual_normalized_loss = torch.sum(size_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # 3.4 Semantic cls loss
    sem_cls_label = torch.gather(data_dict['sem_cls_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
    sem_cls_loss = criterion_sem_cls(data_dict['sem_cls_scores'].transpose(2,1), sem_cls_label) # (B,K)
    sem_cls_loss = torch.sum(sem_cls_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    return center_loss, heading_class_loss, heading_residual_normalized_loss, size_class_loss, size_residual_normalized_loss, sem_cls_loss
    #return center_loss, heading_class_loss, heading_residual_normalized_loss, size_residual_normalized_loss, sem_cls_loss

def compute_reference_loss(data_dict, config):
    """ Compute cluster reference loss

    Args:
        data_dict: dict (read-only)

    Returns:
        ref_loss, lang_loss, cluster_preds, cluster_labels
    """

    # unpack
    cluster_preds = data_dict["cluster_ref"] # (B, num_proposal)

    # predicted bbox
    pred_ref = data_dict['cluster_ref'].detach().cpu().numpy() # (B,)
    pred_center = data_dict['center'].detach().cpu().numpy() # (B,K,3)
    pred_heading_class = torch.argmax(data_dict['heading_scores'], -1) # B,num_proposal
    pred_heading_residual = torch.gather(data_dict['heading_residuals'], 2, pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
    pred_heading_class = pred_heading_class.detach().cpu().numpy() # B,num_proposal
    pred_heading_residual = pred_heading_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal
    pred_size_class = torch.argmax(data_dict['size_scores'], -1) # B,num_proposal
    pred_size_residual = torch.gather(data_dict['size_residuals'], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
    pred_size_class = pred_size_class.detach().cpu().numpy()
    pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal,3

    # ground truth bbox
    gt_center = data_dict['ref_center_label'].cpu().numpy() # (B,3)
    gt_heading_class = data_dict['ref_heading_class_label'].cpu().numpy() # B
    gt_heading_residual = data_dict['ref_heading_residual_label'].cpu().numpy() # B
    gt_size_class = data_dict['ref_size_class_label'].cpu().numpy() # B
    gt_size_residual = data_dict['ref_size_residual_label'].cpu().numpy() # B,3
    # convert gt bbox parameters to bbox corners
    gt_obb_batch = config.param2obb_batch(gt_center[:, 0:3], gt_heading_class, gt_heading_residual,
                    gt_size_class, gt_size_residual)
    gt_bbox_batch = get_3d_box_batch(gt_obb_batch[:, 3:6], gt_obb_batch[:, 6], gt_obb_batch[:, 0:3])

    # compute the iou score for all predictd positive ref
    batch_size, num_proposals = cluster_preds.shape
    labels = np.zeros((batch_size, num_proposals))
    for i in range(pred_ref.shape[0]):
        # convert the bbox parameters to bbox corners
        pred_obb_batch = config.param2obb_batch(pred_center[i, :, 0:3], pred_heading_class[i], pred_heading_residual[i],
                    pred_size_class[i], pred_size_residual[i])
        pred_bbox_batch = get_3d_box_batch(pred_obb_batch[:, 3:6], pred_obb_batch[:, 6], pred_obb_batch[:, 0:3])
        ious = box3d_iou_batch(pred_bbox_batch, np.tile(gt_bbox_batch[i], (num_proposals, 1, 1)))
        labels[i, ious.argmax()] = 1 # treat the bbox with highest iou score as the gt

    cluster_labels = torch.FloatTensor(labels).cuda()

    # reference loss
    criterion = SoftmaxRankingLoss()
    loss = criterion(cluster_preds, cluster_labels.float().clone())

    return loss, cluster_preds, cluster_labels

def compute_lang_classification_loss(data_dict):
    criterion = torch.nn.CrossEntropyLoss()
    # chunking
    if "object_cat_list" in data_dict:
        object_cat_list = data_dict["object_cat_list"]
        batch_size, len_nun_max = object_cat_list.shape[:2]
        lang_num = data_dict["lang_num"]
        lang_scores = data_dict["lang_scores"].reshape(batch_size, len_nun_max, -1)
        loss = 0.
        for i in range(batch_size):
            num = lang_num[i]
            loss += criterion(lang_scores[i, :num], object_cat_list[i, :num])
        loss = loss / batch_size
    else:
        loss = criterion(data_dict["lang_scores"], data_dict["object_cat"])

    return loss
'''
# 3DETR Functions:
class Matcher(nn.Module):
    def __init__(self, cost_class, cost_objectness, cost_giou, cost_center):
        """
        Parameters:
            cost_class:
        Returns:
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_objectness = cost_objectness
        self.cost_giou = cost_giou
        self.cost_center = cost_center

    @torch.no_grad()
    def forward(self, outputs, data_dict):

        batchsize = outputs["sem_cls_prob"].shape[0]
        nqueries = outputs["sem_cls_prob"].shape[1]
        ngt = data_dict["gt_box_sem_cls_label"].shape[1]
        nactual_gt = data_dict["nactual_gt"]

        # classification cost: batch x nqueries x ngt matrix
        pred_cls_prob = outputs["sem_cls_prob"]
        gt_box_sem_cls_labels = (
            data_dict["gt_box_sem_cls_label"]
            .unsqueeze(1)
            .expand(batchsize, nqueries, ngt)
        )
        class_mat = -torch.gather(pred_cls_prob, 2, gt_box_sem_cls_labels)

        # objectness cost: batch x nqueries x 1
        objectness_mat = -outputs["objectness_prob"].unsqueeze(-1)

        # center cost: batch x nqueries x ngt
        center_mat = outputs["center_dist"].detach()

        # giou cost: batch x nqueries x ngt
        giou_mat = -outputs["gious"].detach()

        final_cost = (
            self.cost_class * class_mat
            + self.cost_objectness * objectness_mat
            + self.cost_center * center_mat
            + self.cost_giou * giou_mat
        )

        final_cost = final_cost.detach().cpu().numpy()
        assignments = []

        # auxiliary variables useful for batched loss computation
        batch_size, nprop = final_cost.shape[0], final_cost.shape[1]
        per_prop_gt_inds = torch.zeros(
            [batch_size, nprop], dtype=torch.int64, device=pred_cls_prob.device
        )
        proposal_matched_mask = torch.zeros(
            [batch_size, nprop], dtype=torch.float32, device=pred_cls_prob.device
        )
        for b in range(batchsize):
            assign = []
            if nactual_gt[b] > 0:
                assign = linear_sum_assignment(final_cost[b, :, : nactual_gt[b]])
                assign = [
                    torch.from_numpy(x).long().to(device=pred_cls_prob.device)
                    for x in assign
                ]
                per_prop_gt_inds[b, assign[0]] = assign[1]
                proposal_matched_mask[b, assign[0]] = 1
            assignments.append(assign)

        return {
            "assignments": assignments,
            "per_prop_gt_inds": per_prop_gt_inds,
            "proposal_matched_mask": proposal_matched_mask,
        }
        

def loss_sem_cls(self, outputs, data_dict, assignments):

        # # Not vectorized version
        # pred_logits = outputs["sem_cls_logits"]
        # assign = assignments["assignments"]

        # sem_cls_targets = torch.ones((pred_logits.shape[0], pred_logits.shape[1]),
        #                         dtype=torch.int64, device=pred_logits.device)

        # # initialize to background/no-object class
        # sem_cls_targets *= (pred_logits.shape[-1] - 1)

        # # use assignments to compute labels for matched boxes
        # for b in range(pred_logits.shape[0]):
        #     if len(assign[b]) > 0:
        #         sem_cls_targets[b, assign[b][0]] = targets["gt_box_sem_cls_label"][b, assign[b][1]]

        # sem_cls_targets = sem_cls_targets.view(-1)
        # pred_logits = pred_logits.reshape(sem_cls_targets.shape[0], -1)
        # loss = F.cross_entropy(pred_logits, sem_cls_targets, self.semcls_percls_weights, reduction="mean")

        pred_logits = outputs["sem_cls_logits"]
        gt_box_label = torch.gather(
            data_dict["gt_box_sem_cls_label"], 1, assignments["per_prop_gt_inds"]
        )
        gt_box_label[assignments["proposal_matched_mask"].int() == 0] = (
            pred_logits.shape[-1] - 1
        )
        loss = F.cross_entropy(
            pred_logits.transpose(2, 1),
            gt_box_label,
            self.semcls_percls_weights,
            reduction="mean",
        )

        return {"loss_sem_cls": loss}

def loss_angle(self, outputs, data_dict, assignments):
    angle_logits = outputs["angle_logits"]
    angle_residual = outputs["angle_residual_normalized"]

    if data_dict["num_boxes_replica"] > 0:
        gt_angle_label = data_dict["gt_angle_class_label"]
        gt_angle_residual = data_dict["gt_angle_residual_label"]
        gt_angle_residual_normalized = gt_angle_residual / (
            np.pi / self.dataset_config.num_angle_bin
        )

        # # Non vectorized version
        # assignments = assignments["assignments"]
        # p_angle_logits = []
        # p_angle_resid = []
        # t_angle_labels = []
        # t_angle_resid = []

        # for b in range(angle_logits.shape[0]):
        #     if len(assignments[b]) > 0:
        #         p_angle_logits.append(angle_logits[b, assignments[b][0]])
        #         p_angle_resid.append(angle_residual[b, assignments[b][0], gt_angle_label[b][assignments[b][1]]])
        #         t_angle_labels.append(gt_angle_label[b, assignments[b][1]])
        #         t_angle_resid.append(gt_angle_residual_normalized[b, assignments[b][1]])

        # p_angle_logits = torch.cat(p_angle_logits)
        # p_angle_resid = torch.cat(p_angle_resid)
        # t_angle_labels = torch.cat(t_angle_labels)
        # t_angle_resid = torch.cat(t_angle_resid)

        # angle_cls_loss = F.cross_entropy(p_angle_logits, t_angle_labels, reduction="sum")
        # angle_reg_loss = huber_loss(p_angle_resid.flatten() - t_angle_resid.flatten()).sum()

        gt_angle_label = torch.gather(
            gt_angle_label, 1, assignments["per_prop_gt_inds"]
        )
        angle_cls_loss = F.cross_entropy(
            angle_logits.transpose(2, 1), gt_angle_label, reduction="none"
        )
        angle_cls_loss = (
            angle_cls_loss * assignments["proposal_matched_mask"]
        ).sum()

        gt_angle_residual_normalized = torch.gather(
            gt_angle_residual_normalized, 1, assignments["per_prop_gt_inds"]
        )
        gt_angle_label_one_hot = torch.zeros_like(
            angle_residual, dtype=torch.float32
        )
        gt_angle_label_one_hot.scatter_(2, gt_angle_label.unsqueeze(-1), 1)

        angle_residual_for_gt_class = torch.sum(
            angle_residual * gt_angle_label_one_hot, -1
        )
        angle_reg_loss = huber_loss(
            angle_residual_for_gt_class - gt_angle_residual_normalized, delta=1.0
        )
        angle_reg_loss = (
            angle_reg_loss * assignments["proposal_matched_mask"]
        ).sum()

        angle_cls_loss /= data_dict["num_boxes"]
        angle_reg_loss /= data_dict["num_boxes"]
    else:
        angle_cls_loss = torch.zeros(1, device=angle_logits.device).squeeze()
        angle_reg_loss = torch.zeros(1, device=angle_logits.device).squeeze()
    return {"loss_angle_cls": angle_cls_loss, "loss_angle_reg": angle_reg_loss}

def loss_center(self, outputs, data_dict, assignments):
    center_dist = outputs["center_dist"]
    if data_dict["num_boxes_replica"] > 0:

        # # Non vectorized version
        # assign = assignments["assignments"]
        # center_loss = torch.zeros(1, device=center_dist.device).squeeze()
        # for b in range(center_dist.shape[0]):
        #     if len(assign[b]) > 0:
        #         center_loss += center_dist[b, assign[b][0], assign[b][1]].sum()

        # select appropriate distances by using proposal to gt matching
        center_loss = torch.gather(
            center_dist, 2, assignments["per_prop_gt_inds"].unsqueeze(-1)
        ).squeeze(-1)
        # zero-out non-matched proposals
        center_loss = center_loss * assignments["proposal_matched_mask"]
        center_loss = center_loss.sum()

        if data_dict["num_boxes"] > 0:
            center_loss /= data_dict["num_boxes"]
    else:
        center_loss = torch.zeros(1, device=center_dist.device).squeeze()

    return {"loss_center": center_loss}

def loss_giou(self, outputs, data_dict, assignments):
    gious_dist = 1 - outputs["gious"]

    # # Non vectorized version
    # giou_loss = torch.zeros(1, device=gious_dist.device).squeeze()
    # assign = assignments["assignments"]

    # for b in range(gious_dist.shape[0]):
    #     if len(assign[b]) > 0:
    #         giou_loss += gious_dist[b, assign[b][0], assign[b][1]].sum()

    # select appropriate gious by using proposal to gt matching
    giou_loss = torch.gather(
        gious_dist, 2, assignments["per_prop_gt_inds"].unsqueeze(-1)
    ).squeeze(-1)
    # zero-out non-matched proposals
    giou_loss = giou_loss * assignments["proposal_matched_mask"]
    giou_loss = giou_loss.sum()

    if data_dict["num_boxes"] > 0:
        giou_loss /= data_dict["num_boxes"]

    return {"loss_giou": giou_loss}

def loss_size(self, outputs, data_dict, assignments):
    gt_box_sizes = data_dict["gt_box_sizes_normalized"]
    pred_box_sizes = outputs["size_normalized"]

    if data_dict["num_boxes_replica"] > 0:

        # # Non vectorized version
        # p_sizes = []
        # t_sizes = []
        # assign = assignments["assignments"]
        # for b in range(pred_box_sizes.shape[0]):
        #     if len(assign[b]) > 0:
        #         p_sizes.append(pred_box_sizes[b, assign[b][0]])
        #         t_sizes.append(gt_box_sizes[b, assign[b][1]])
        # p_sizes = torch.cat(p_sizes)
        # t_sizes = torch.cat(t_sizes)
        # size_loss = F.l1_loss(p_sizes, t_sizes, reduction="sum")

        # construct gt_box_sizes as [batch x nprop x 3] matrix by using proposal to gt matching
        gt_box_sizes = torch.stack(
            [
                torch.gather(
                    gt_box_sizes[:, :, x], 1, assignments["per_prop_gt_inds"]
                )
                for x in range(gt_box_sizes.shape[-1])
            ],
            dim=-1,
        )
        size_loss = F.l1_loss(pred_box_sizes, gt_box_sizes, reduction="none").sum(
            dim=-1
        )

        # zero-out non-matched proposals
        size_loss *= assignments["proposal_matched_mask"]
        size_loss = size_loss.sum()

        size_loss /= data_dict["num_boxes"]
    else:
        size_loss = torch.zeros(1, device=pred_box_sizes.device).squeeze()
    return {"loss_size": size_loss}

# Define matcher:
matcher = Matcher(0,0,0,0)

def single_output_forward(self, outputs, data_dict):
    gious = generalized_box3d_iou(
        outputs["box_corners"],
        data_dict["gt_box_corners"],
        data_dict["nactual_gt"],
        rotated_boxes=torch.any(data_dict["gt_box_angles"] > 0).item(),
        needs_grad=(self.loss_weight_dict["loss_giou_weight"] > 0),
    )

    outputs["gious"] = gious
    center_dist = torch.cdist(
        outputs["center_normalized"], data_dict["gt_box_centers_normalized"], p=1
    )
    outputs["center_dist"] = center_dist
    assignments = matcher(outputs, data_dict)

    losses = {}

    # change it --> not working yet
    for k in self.loss_functions:
        loss_wt_key = k + "_weight"
        if (
            loss_wt_key in self.loss_weight_dict
            and self.loss_weight_dict[loss_wt_key] > 0
        ) or loss_wt_key not in self.loss_weight_dict:
            # only compute losses with loss_wt > 0
            # certain losses like cardinality are only logged and have no loss weight
            curr_loss = self.loss_functions[k](data_dict, data_dict, assignments)
            losses.update(curr_loss)

    final_loss = 0
    for k in self.loss_weight_dict:
        if self.loss_weight_dict[k] > 0:
            losses[k.replace("_weight", "")] *= self.loss_weight_dict[k]
            final_loss += losses[k.replace("_weight", "")]
    return final_loss, losses

def forward(self, data_dict):
    nactual_gt = data_dict["gt_box_present"].sum(axis=1).long()
    num_boxes = torch.clamp(all_reduce_average(nactual_gt.sum()), min=1).item()
    data_dict["nactual_gt"] = nactual_gt
    data_dict["num_boxes"] = num_boxes
    data_dict[
        "num_boxes_replica"
    ] = nactual_gt.sum().item()  # number of boxes on this worker for dist training

    loss, loss_dict = self.single_output_forward(data_dict['outputs'], data_dict)    
    if "aux_outputs" in data_dict:
        for k in range(len(data_dict["aux_outputs"])):
            interm_loss, interm_loss_dict = self.single_output_forward(
                data_dict["aux_outputs"][k], data_dict
            )

            loss += interm_loss
            for interm_key in interm_loss_dict:
                loss_dict[f"{interm_key}_{k}"] = interm_loss_dict[interm_key]
    
    return loss, loss_dict
'''
def get_loss(data_dict, config, detection=True, reference=True, use_lang_classifier=True):
    """ Loss functions

    Args:
        data_dict: dict
        config: dataset config instance
        reference: flag (False/True)
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """
    '''
    # Vote loss
    vote_loss = compute_vote_loss(data_dict)

    # Obj loss
    objectness_loss, objectness_label, objectness_mask, object_assignment = compute_objectness_loss(data_dict)
    num_proposal = objectness_label.shape[1]
    total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]
    data_dict['objectness_label'] = objectness_label
    data_dict['objectness_mask'] = objectness_mask
    data_dict['object_assignment'] = object_assignment
    data_dict['pos_ratio'] = torch.sum(objectness_label.float().cuda())/float(total_num_proposal)
    data_dict['neg_ratio'] = torch.sum(objectness_mask.float())/float(total_num_proposal) - data_dict['pos_ratio']

    # Box loss and sem cls loss
    center_loss, heading_cls_loss, heading_reg_loss, size_cls_loss, size_reg_loss, sem_cls_loss = compute_box_and_sem_cls_loss(data_dict, config)
    box_loss = center_loss + 0.1*heading_cls_loss + heading_reg_loss + 0.1*size_cls_loss + size_reg_loss
    #center_loss, heading_cls_loss, heading_reg_loss, size_reg_loss, sem_cls_loss = compute_box_and_sem_cls_loss(data_dict, config)
    #box_loss = center_loss + 0.1*heading_cls_loss + heading_reg_loss + size_reg_loss

    # 3DETR Obj detection loss:
    obj_loss, loss_dict = forward(data_dict)
    
    if detection:
        data_dict['center_loss'] = loss_dict['loss_center']
        data_dict['size_loss'] = loss_dict['loss_size']
        data_dict['giou_loss'] = loss_dict['loss_giou']
        data_dict['angle_cls_loss'] = loss_dict['loss_angle_cls']
        data_dict['angle_reg_loss'] = loss_dict['loss_angle_reg']
        data_dict['sem_cls_loss'] = loss_dict['loss_sem_cls']
        data_dict['obj_loss'] = obj_loss
    else:
        data_dict['center_loss'] = torch.zeros(1)[0].cuda()
        data_dict['size_loss'] = torch.zeros(1)[0].cuda()
        data_dict['giou_loss'] = torch.zeros(1)[0].cuda()
        data_dict['angle_cls_loss'] = torch.zeros(1)[0].cuda()
        data_dict['angle_reg_loss'] = torch.zeros(1)[0].cuda()
        data_dict['sem_cls_loss'] = torch.zeros(1)[0].cuda()
        data_dict['obj_loss'] = torch.zeros(1)[0].cuda()
    '''
    detection = False
    if detection:
        data_dict['vote_loss'] = vote_loss
        data_dict['objectness_loss'] = objectness_loss
        data_dict['center_loss'] = center_loss
        data_dict['heading_cls_loss'] = heading_cls_loss
        data_dict['heading_reg_loss'] = heading_reg_loss
        data_dict['size_cls_loss'] = size_cls_loss
        data_dict['size_reg_loss'] = size_reg_loss
        data_dict['sem_cls_loss'] = sem_cls_loss
        data_dict['box_loss'] = box_loss
    else:
        data_dict['vote_loss'] = torch.zeros(1)[0].cuda()
        data_dict['objectness_loss'] = torch.zeros(1)[0].cuda()
        data_dict['center_loss'] = torch.zeros(1)[0].cuda()
        data_dict['heading_cls_loss'] = torch.zeros(1)[0].cuda()
        data_dict['heading_reg_loss'] = torch.zeros(1)[0].cuda()
        data_dict['size_cls_loss'] = torch.zeros(1)[0].cuda()
        data_dict['size_reg_loss'] = torch.zeros(1)[0].cuda()
        data_dict['sem_cls_loss'] = torch.zeros(1)[0].cuda()
        data_dict['box_loss'] = torch.zeros(1)[0].cuda()
    reference = False
    if reference:
        # Reference loss
        ref_loss, _, cluster_labels = compute_reference_loss(data_dict, config)
        data_dict["cluster_labels"] = cluster_labels
        data_dict["ref_loss"] = ref_loss
    else:
        # # Reference loss
        # ref_loss, _, cluster_labels = compute_reference_loss(data_dict, config)
        # data_dict["cluster_labels"] = cluster_labels
        '''
        data_dict["cluster_labels"] = objectness_label.new_zeros(objectness_label.shape).cuda()
        data_dict["cluster_ref"] = objectness_label.new_zeros(objectness_label.shape).float().cuda()
        '''
        # store
        data_dict["ref_loss"] = torch.zeros(1)[0].cuda()
    reference = True
    if reference and use_lang_classifier:
        data_dict["lang_loss"] = compute_lang_classification_loss(data_dict)
    else:
        data_dict["lang_loss"] = torch.zeros(1)[0].cuda()

    # Final loss function
    '''
    loss = data_dict['vote_loss'] + 0.5*data_dict['objectness_loss'] + data_dict['box_loss'] + 0.1*data_dict['sem_cls_loss'] \
        + 0.1*data_dict["ref_loss"] + 0.1*data_dict["lang_loss"]
    '''
    loss = data_dict["lang_loss"]
    # print(f'data_dict["lang_loss"]: {data_dict["lang_loss"]}')
    # print(f'loss: {loss}')

    loss *= 10 # amplify

    data_dict['loss'] = loss

    return loss, data_dict
