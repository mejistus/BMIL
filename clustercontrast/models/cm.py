import collections
import math
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



class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, mode='CM', smooth=0, has_multi_pos_loss=False):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.cm_type = mode
        self.has_multi_pos_loss = has_multi_pos_loss

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


    def forward(self, inputs, targets):

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
                loss = 0.5 * (self.get_multi_pos_loss(hard, targets) + self.get_multi_pos_loss(mean, targets))
            else:
                r = 0.2
                loss = 0.5 * (self.cross_entropy(hard, targets) + torch.relu(self.cross_entropy(mean, targets) - r))
            return loss


    def get_multi_pos_loss(self, score, batch_proxy_ind):
        assert self.proxy_pseudo_labels is not None
        identity_score, proxy_counts = self._identity_balanced_logits(score)
        target_identity = self.proxy_pseudo_labels[batch_proxy_ind]
        num_identities = identity_score.size(1)
        bg_knn = 50

        if num_identities == 1:
            return identity_score.sum() * 0.0

        positive_score = identity_score.gather(1, target_identity.unsqueeze(1))
        negative_score = identity_score.clone()
        negative_score.scatter_(1, target_identity.unsqueeze(1), float('-inf'))
        negative_count = min(bg_knn - 1, num_identities - 1)
        hard_negative_score = torch.topk(
            negative_score, k=negative_count, dim=1
        ).values
        selected_score = torch.cat([positive_score, hard_negative_score], dim=1)
        targets = torch.zeros(
            selected_score.size(0), dtype=torch.long, device=selected_score.device
        )

        return F.cross_entropy(selected_score, targets)

    def _identity_balanced_logits(self, score):
        """Collapse one or two modality proxies into one equal-mass identity logit."""
        proxy_labels = self.proxy_pseudo_labels
        unique_labels, proxy_counts = torch.unique_consecutive(
            proxy_labels, return_counts=True
        )
        expected_labels = torch.arange(
            len(unique_labels), dtype=unique_labels.dtype, device=unique_labels.device
        )
        if not torch.equal(unique_labels, expected_labels):
            raise ValueError("proxy pseudo labels must be contiguous and identity-sorted")
        if torch.any((proxy_counts < 1) | (proxy_counts > 2)):
            raise ValueError("each identity must have one or two modality proxies")

        starts = torch.cumsum(proxy_counts, dim=0) - proxy_counts
        identity_score = score[:, starts].clone()
        dual_proxy = proxy_counts == 2
        if torch.any(dual_proxy):
            dual_starts = starts[dual_proxy]
            dual_scores = torch.stack(
                [score[:, dual_starts], score[:, dual_starts + 1]], dim=0
            )
            identity_score[:, dual_proxy] = (
                torch.logsumexp(dual_scores, dim=0) - math.log(2.0)
            )

        return identity_score, proxy_counts
