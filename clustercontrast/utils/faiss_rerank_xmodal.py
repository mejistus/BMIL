#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

import os, sys
import time
import numpy as np
from scipy.spatial.distance import cdist
import gc
import faiss

import torch
import torch.nn.functional as F

from .faiss_utils import search_index_pytorch, search_raw_array_pytorch, index_init_gpu, index_init_cpu


def select_local_boundary(similarities, min_k=3, max_k=6, threshold=2.5):
    """Choose a stable local cut after rank k, or fall back to max_k."""
    values = np.asarray(similarities, dtype=np.float64)
    if values.ndim != 1 or len(values) < max_k + 1:
        raise ValueError("similarities must contain at least max_k + 1 ranks")
    if min_k < 1 or min_k > max_k:
        raise ValueError("expected 1 <= min_k <= max_k")

    gaps = values[:max_k] - values[1:max_k + 1]
    reference = gaps[min_k - 1:max_k]
    median = np.median(reference)
    mad = np.median(np.abs(reference - median))
    scale = 1.4826 * mad + 1e-12
    robust_scores = (reference - median) / scale
    best_offset = int(np.argmax(robust_scores))
    if robust_scores[best_offset] < threshold:
        return max_k
    return min_k + best_offset


def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i,:k1+1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
    fi = np.where(backward_k_neigh_index==i)[0]
    return forward_k_neigh_index[fi]


def compute_jaccard_distance_xmodal(target_features, k1=20, k2=6, print_flag=True, search_option=0, use_float16=False, all_file_name=None, rgbNum=0):
    end = time.time()
    if print_flag:
        print('Computing jaccard distance...')

    ngpus = faiss.get_num_gpus()
    N = target_features.size(0)
    mat_type = np.float16 if use_float16 else np.float32

    if (search_option==0):
        # GPU + PyTorch CUDA Tensors (1)
        res = faiss.StandardGpuResources()
        res.setDefaultNullStreamAllDevices()
        _, initial_rank = search_raw_array_pytorch(res, target_features, target_features, k1)
        initial_rank = initial_rank.cpu().numpy()

    elif (search_option==1):
        # GPU + PyTorch CUDA Tensors (2)
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = search_index_pytorch(index, target_features, k1)
        res.syncDefaultStreamCurrentDevice()
        initial_rank = initial_rank.cpu().numpy()

    elif (search_option==2):
        # GPU
        index = index_init_gpu(ngpus, target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = index.search(target_features.cpu().numpy(), k1)

    elif (search_option==3):
        # CPU
        index = index_init_cpu(target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = index.search(target_features.cpu().numpy(), k1)

    elif (search_option>=4):
        # search intra-modality KNN and inter-modality KNN separately
        sim_mat = torch.mm(target_features, target_features.t())
        sim1, rank1 = torch.topk(sim_mat[:, 0:rgbNum], k=k1//2, dim=1)
        sim2, rank2 = torch.topk(sim_mat[:, rgbNum:], k=k1//2, dim=1)
        rank2 += rgbNum
        concated_sim = torch.cat([sim1, sim2], dim=1)
        concated_rank = torch.cat([rank1, rank2], dim=1)
        sorted_inds = torch.argsort(concated_sim, dim=1, descending=True)
        for i in range(len(concated_rank)):
            concated_rank[i] = concated_rank[i, sorted_inds[i]] 
        initial_rank = concated_rank

    nn_k1 = []
    nn_k1_half = []
    for i in range(N):
        nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1))
        nn_k1_half.append(k_reciprocal_neigh(initial_rank, i, int(np.around(k1/2))))

    V = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]
            if (len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index)) > 2/3*len(candidate_k_reciprocal_index)):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  ## element-wise unique
        dist = 2-2*torch.mm(target_features[i].unsqueeze(0).contiguous(), target_features[k_reciprocal_expansion_index].t())
        if use_float16:
            V[i,k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy().astype(mat_type)
        else:
            V[i,k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy()

    del nn_k1, nn_k1_half

    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=mat_type)
        selected_vis_k = []
        selected_ir_k = []
        for i in range(N):
            #V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:], axis=0)
            if search_option >= 4:
                vis_k = select_local_boundary(sim1[i].cpu().numpy(), max_k=k2)
                ir_k = select_local_boundary(sim2[i].cpu().numpy(), max_k=k2)
                selected_vis_k.append(vis_k)
                selected_ir_k.append(ir_k)
                vis_mean = np.mean(V[rank1[i,:vis_k],:], axis=0)
                ir_mean = np.mean(V[rank2[i,:ir_k],:], axis=0)
                V_qe[i, :] = np.mean([vis_mean, ir_mean], axis=0)
            else:
                feas_NIR_temp, feas_VIS_temp = [], []
                for ii in initial_rank[i, :k2]:
                    if 'ir_modify' in all_file_name[ii]:
                        feas_NIR_temp.append(V[ii, :])
                    elif 'rgb_modify' in all_file_name[ii]:
                        feas_VIS_temp.append(V[ii, :])
                feas_VIS_temp = np.array(feas_VIS_temp)
                feas_NIR_temp = np.array(feas_NIR_temp)

                NIR_embed_mean = np.mean(feas_NIR_temp, axis=0)
                VIS_embed_mean = np.mean(feas_VIS_temp, axis=0)
                if len(feas_VIS_temp) == 0:
                    V_qe[i, :] = NIR_embed_mean
                elif len(feas_NIR_temp) == 0:
                    V_qe[i, :] = VIS_embed_mean
                else:
                    V_qe[i, :] = np.mean([NIR_embed_mean, VIS_embed_mean], axis=0)
        V = V_qe
        del V_qe
        if print_flag and search_option >= 4:
            print(
                "Local boundary LQE: visible mean {:.2f}, infrared mean {:.2f}, "
                "fallback rates {:.2%}/{:.2%}".format(
                    np.mean(selected_vis_k), np.mean(selected_ir_k),
                    np.mean(np.asarray(selected_vis_k) == k2),
                    np.mean(np.asarray(selected_ir_k) == k2),
                )
            )

    del initial_rank

    invIndex = []
    for i in range(N):
        invIndex.append(np.where(V[:,i] != 0)[0])  #len(invIndex)=all_num

    jaccard_dist = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        temp_min = np.zeros((1, N), dtype=mat_type)
        # temp_max = np.zeros((1,N), dtype=mat_type)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]]+np.minimum(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
            # temp_max[0,indImages[j]] = temp_max[0,indImages[j]]+np.maximum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])

        jaccard_dist[i] = 1-temp_min/(2-temp_min)
        # jaccard_dist[i] = 1-temp_min/(temp_max+1e-6)

    del invIndex, V

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0
    if print_flag:
        print("Jaccard distance computing time cost: {}".format(time.time()-end))

    return jaccard_dist
