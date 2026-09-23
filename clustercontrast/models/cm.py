import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd
from .losses import CrossEntropyLabelSmooth
from IPython import embed


class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hybrid(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        nums = len(ctx.features)//2
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            mean = torch.stack(features, dim=0).mean(0)
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * mean
            ctx.features[index] /= ctx.features[index].norm()
            
            hard = np.argmin(np.array(distances))
            ctx.features[index+nums] = ctx.features[index+nums] * ctx.momentum + (1 - ctx.momentum) * features[hard]
            ctx.features[index+nums] /= ctx.features[index+nums].norm()


        return grad_inputs, None, None, None


def cm_hybrid(inputs, indexes, features, momentum=0.5):
    return CM_Hybrid.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


@torch.no_grad()
def cmcra_relation_weights(masked_features, target_features, targets,
                           proxy_features, proxy_labels, scale=0.1,
                           min_weight=0.05):
    """Measure opposite-proxy instability under complementary masks."""
    if masked_features.ndim != 3 or masked_features.size(1) % 2 != 0:
        raise ValueError("masked_features must contain complementary view pairs")
    if scale <= 0:
        raise ValueError("CMCRA scale must be positive")

    masked_features = F.normalize(masked_features.detach(), dim=2)
    target_features = F.normalize(target_features.detach(), dim=1)
    proxy_features = F.normalize(proxy_features.detach(), dim=1)
    weights = target_features.new_ones(targets.shape[0])
    instability = target_features.new_zeros(targets.shape[0])
    matching = proxy_labels[targets].unsqueeze(1) == proxy_labels.unsqueeze(0)
    matching.scatter_(1, targets.unsqueeze(1), False)
    valid = matching.sum(dim=1) == 1
    opposite_indices = matching.to(torch.int64).argmax(dim=1)

    own_proxy = proxy_features[targets]
    opposite_proxy = proxy_features[opposite_indices]
    base_margin = (
        (target_features * opposite_proxy).sum(dim=1)
        - (target_features * own_proxy).sum(dim=1)
    )
    masked_margin = (
        (masked_features * opposite_proxy.unsqueeze(1)).sum(dim=2)
        - (masked_features * own_proxy.unsqueeze(1)).sum(dim=2)
    )
    drops = torch.relu(base_margin.unsqueeze(1) - masked_margin)
    pair_worst = drops.view(drops.size(0), -1, 2).max(dim=2).values
    relation_instability = pair_worst.median(dim=1).values
    instability[valid] = relation_instability[valid]
    weights[valid] = torch.exp(
        -relation_instability[valid] / scale
    ).clamp_min(min_weight)

    return weights, instability, valid


def weighted_multi_positive_loss(score, batch_proxy_ind, proxy_labels,
                                 relation_weights=None, bg_knn=50):
    """BMIL multi-positive loss with an optional opposite-positive gate.

    A unit gate is exactly the released BMIL objective. A zero gate removes the
    opposite-modality positive from both the target and the normalizer.
    """
    positive_mask = (
        proxy_labels[batch_proxy_ind].unsqueeze(1)
        == proxy_labels.unsqueeze(0)
    )
    ranking_score = score.detach().masked_fill(positive_mask, 10000)
    selected_indices = torch.topk(
        ranking_score, k=min(bg_knn, score.size(1)), dim=1
    ).indices
    selected_score = score.gather(1, selected_indices)
    selected_positive = positive_mask.gather(1, selected_indices)
    selected_own = selected_indices == batch_proxy_ind.unsqueeze(1)
    selected_opposite = selected_positive & ~selected_own

    if relation_weights is None:
        relation_weights = score.new_ones(score.size(0))
    else:
        relation_weights = relation_weights.to(score)
    positive_weights = (
        selected_own.to(score.dtype)
        + selected_opposite.to(score.dtype) * relation_weights.unsqueeze(1)
    )
    log_weights = torch.where(
        relation_weights > 0,
        relation_weights.log(),
        relation_weights.new_full(relation_weights.shape, float("-inf")),
    )
    denominator_score = selected_score + torch.where(
        selected_opposite,
        log_weights.unsqueeze(1),
        torch.zeros_like(selected_score),
    )
    target_score = (
        selected_score * positive_weights
    ).sum(dim=1) / positive_weights.sum(dim=1)
    return (
        torch.logsumexp(denominator_score, dim=1) - target_score
    ).mean()



class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, mode='CM', smooth=0, has_multi_pos_loss=False):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.cm_type = mode
        self.has_multi_pos_loss = has_multi_pos_loss
        self.last_cmcra_stats = None

        if smooth > 0:
            self.cross_entropy = CrossEntropyLabelSmooth(self.num_samples, 0.1, True)
            print('>>> Using CrossEntropy with Label Smoothing.')
        else:
            self.cross_entropy = nn.CrossEntropyLoss().cuda()

        if self.cm_type == 'CM':
            self.register_buffer('features', torch.zeros(num_samples, num_features))
        elif self.cm_type == 'CMhybrid':
            self.register_buffer('features', torch.zeros(2 * num_samples, num_features))           
        else:
            raise TypeError('Cluster Memory {} is invalid!'.format(self.cm_type))


    def forward(self, inputs, targets, masked_features=None,
                target_features=None, cmcra_scale=0.1,
                cmcra_min_weight=0.05):

        inputs = F.normalize(inputs, dim=1).cuda()
        if self.cm_type == 'CM':
            outputs = cm(inputs, targets, self.features, self.momentum)
            outputs /= self.temp
            loss = self.cross_entropy(outputs, targets)
            return loss

        elif self.cm_type == 'CMhybrid':
            outputs = cm_hybrid(inputs, targets, self.features, self.momentum)
            outputs /= self.temp
            mean, hard = torch.chunk(outputs, 2, dim=1)

            if self.has_multi_pos_loss:
                relation_weights = None
                self.last_cmcra_stats = None
                if masked_features is not None:
                    if target_features is None:
                        raise ValueError("CMCRA requires unmasked target features")
                    relation_weights, instability, valid = cmcra_relation_weights(
                        masked_features,
                        target_features,
                        targets,
                        self.features[:len(self.proxy_pseudo_labels)],
                        self.proxy_pseudo_labels,
                        scale=cmcra_scale,
                        min_weight=cmcra_min_weight,
                    )
                    valid_weights = relation_weights[valid]
                    valid_instability = instability[valid]
                    self.last_cmcra_stats = torch.stack((
                        valid.sum().to(valid_weights.dtype),
                        valid_weights.sum(),
                        valid_instability.sum(),
                        (valid_weights < 0.5).sum().to(valid_weights.dtype),
                    )).detach()
                loss = 0.5 * (
                    self.get_multi_pos_loss(hard, targets, relation_weights)
                    + self.get_multi_pos_loss(mean, targets, relation_weights)
                )
            else:
                r = 0.2
                loss = 0.5 * (self.cross_entropy(hard, targets) + torch.relu(self.cross_entropy(mean, targets) - r))
            return loss


    def get_multi_pos_loss(self, score, batch_proxy_ind,
                           relation_weights=None):
        assert(self.proxy_pseudo_labels is not None)
        return weighted_multi_positive_loss(
            score,
            batch_proxy_ind,
            self.proxy_pseudo_labels,
            relation_weights=relation_weights,
        )
