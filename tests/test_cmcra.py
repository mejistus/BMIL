import torch
import torch.nn.functional as F

from clustercontrast.models.agw import complementary_block_masks
from clustercontrast.models.cm import (
    cmcra_relation_weights,
    weighted_multi_positive_loss,
)


def released_multi_positive_loss(score, targets, proxy_labels, bg_knn=50):
    ranking_score = score.detach().clone()
    loss = score.new_zeros(())
    for row in range(len(score)):
        pseudo_label = proxy_labels[targets[row]]
        positives = torch.nonzero(proxy_labels == pseudo_label).squeeze(-1)
        ranking_score[row, positives] = 10000
        selected = torch.topk(ranking_score[row], k=bg_knn).indices
        selected_score = score[row, selected]
        selected_target = torch.zeros_like(selected_score)
        selected_target[:len(positives)] = 1.0 / len(positives)
        loss -= (F.log_softmax(selected_score, dim=0) * selected_target).sum()
    return loss / len(score)


def proxy_labels():
    labels = torch.arange(60)
    labels[0:2] = 100
    labels[2:4] = 101
    labels[4] = 102
    return labels


def test_unit_gate_matches_released_bmil_loss():
    torch.manual_seed(7)
    score = torch.randn(3, 60)
    targets = torch.tensor([0, 2, 4])
    labels = proxy_labels()
    expected = released_multi_positive_loss(score, targets, labels)
    actual = weighted_multi_positive_loss(
        score, targets, labels, relation_weights=torch.ones(3)
    )
    torch.testing.assert_close(actual, expected)


def test_zero_gate_removes_opposite_positive_from_normalizer():
    torch.manual_seed(11)
    score = torch.randn(1, 60)
    targets = torch.tensor([0])
    labels = proxy_labels()
    actual = weighted_multi_positive_loss(
        score, targets, labels, relation_weights=torch.zeros(1)
    )

    ranking = score[0].detach().clone()
    ranking[0:2] = 10000
    selected = torch.topk(ranking, k=50).indices
    selected_score = score[0, selected]
    selected_score[selected == 1] = float("-inf")
    expected = torch.logsumexp(selected_score, dim=0) - score[0, 0]
    torch.testing.assert_close(actual, expected)


def test_relation_gate_uses_relative_complementary_mask_drop():
    proxies = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([3, 3])
    targets = torch.tensor([0, 0])
    unmasked = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    stable = torch.tensor([0.0, 1.0]).repeat(6, 1)
    brittle = torch.tensor([1.0, 0.0]).repeat(6, 1)
    masked = torch.stack((stable, brittle), dim=0)

    weights, instability, valid = cmcra_relation_weights(
        masked,
        unmasked,
        targets,
        proxies,
        labels,
        scale=0.1,
        min_weight=0.05,
    )

    assert valid.tolist() == [True, True]
    torch.testing.assert_close(weights[0], torch.tensor(1.0))
    torch.testing.assert_close(instability[0], torch.tensor(0.0))
    torch.testing.assert_close(weights[1], torch.tensor(0.05))
    torch.testing.assert_close(instability[1], torch.tensor(2.0))


def test_complementary_masks_have_equal_area():
    torch.manual_seed(13)
    spatial = torch.ones(4, 8, 18, 9)
    masks = complementary_block_masks(spatial, num_pairs=3, block_size=3)

    assert masks.shape == (4, 6, 1, 18, 9)
    pair_sums = masks.view(4, 3, 2, 1, 18, 9).sum(dim=2)
    torch.testing.assert_close(pair_sums, torch.ones_like(pair_sums))
    visible_area = masks.sum(dim=(2, 3, 4))
    torch.testing.assert_close(
        visible_area, torch.full_like(visible_area, 81.0)
    )
