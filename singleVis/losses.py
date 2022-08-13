import torch
from torch import nn
import torch.nn.functional as F
from singleVis.backend import convert_distance_to_probability, compute_cross_entropy

"""Losses modules for preserving four propertes"""
# https://github.com/ynjnpa/VocGAN/blob/5339ee1d46b8337205bec5e921897de30a9211a1/utils/stft_loss.py for losses module


class UmapLoss(nn.Module):
    def __init__(self, negative_sample_rate, device, _a=1.0, _b=1.0, repulsion_strength=1.0):
        super(UmapLoss, self).__init__()

        self._negative_sample_rate = negative_sample_rate
        self._a = _a,
        self._b = _b,
        self._repulsion_strength = repulsion_strength
        self.DEVICE = torch.device(device)

    @property
    def a(self):
        return self._a[0]

    @property
    def b(self):
        return self._b[0]

    def forward(self, embedding_to, embedding_from):
        batch_size = embedding_to.shape[0]
        # get negative samples
        embedding_neg_to = torch.repeat_interleave(embedding_to, self._negative_sample_rate, dim=0)
        repeat_neg = torch.repeat_interleave(embedding_from, self._negative_sample_rate, dim=0)
        randperm = torch.randperm(repeat_neg.shape[0])
        embedding_neg_from = repeat_neg[randperm]

        #  distances between samples (and negative samples)
        distance_embedding = torch.cat(
            (
                torch.norm(embedding_to - embedding_from, dim=1),
                torch.norm(embedding_neg_to - embedding_neg_from, dim=1),
            ),
            dim=0,
        )
        probabilities_distance = convert_distance_to_probability(
            distance_embedding, self.a, self.b
        )
        probabilities_distance = probabilities_distance.to(self.DEVICE)

        # set true probabilities based on negative sampling
        probabilities_graph = torch.cat(
            (torch.ones(batch_size), torch.zeros(batch_size * self._negative_sample_rate)), dim=0,
        )
        probabilities_graph = probabilities_graph.to(device=self.DEVICE)

        # compute cross entropy
        (_, _, ce_loss) = compute_cross_entropy(
            probabilities_graph,
            probabilities_distance,
            repulsion_strength=self._repulsion_strength,
        )

        return torch.mean(ce_loss)


class ReconstructionLoss(nn.Module):
    def __init__(self, beta=1.0):
        super(ReconstructionLoss, self).__init__()
        self._beta = beta

    def forward(self, edge_to, edge_from, recon_to, recon_from, a_to, a_from):
        loss1 = torch.mean(torch.mean(torch.multiply(torch.pow((1+a_to), self._beta), torch.pow(edge_to - recon_to, 2)), 1))
        loss2 = torch.mean(torch.mean(torch.multiply(torch.pow((1+a_from), self._beta), torch.pow(edge_from - recon_from, 2)), 1))
        # without attention weights
        # loss1 = torch.mean(torch.mean(torch.pow(edge_to - recon_to, 2), 1))
        # loss2 = torch.mean(torch.mean(torch.pow(edge_from - recon_from, 2), 1))
        return (loss1 + loss2)/2


class SmoothnessLoss(nn.Module):
    def __init__(self, margin=0.0):
        super(SmoothnessLoss, self).__init__()
        self._margin = margin

    def forward(self, embedding, target, Coefficient):
        loss = torch.mean(Coefficient * torch.clamp(torch.norm(embedding-target, dim=1)-self._margin, min=0))
        return loss


class SingleVisLoss(nn.Module):
    def __init__(self, umap_loss, recon_loss, lambd):
        super(SingleVisLoss, self).__init__()
        self.umap_loss = umap_loss
        self.recon_loss = recon_loss
        self.lambd = lambd

    def forward(self, edge_to, edge_from, a_to, a_from, outputs):
        embedding_to, embedding_from = outputs["umap"]
        recon_to, recon_from = outputs["recon"]

        recon_l = self.recon_loss(edge_to, edge_from, recon_to, recon_from, a_to, a_from)
        # recon_l = self.recon_loss(edge_to, edge_from, recon_to, recon_from)
        umap_l = self.umap_loss(embedding_to, embedding_from)

        loss = umap_l + self.lambd * recon_l

        return umap_l, recon_l, loss

class HybridLoss(nn.Module):
    def __init__(self, umap_loss, recon_loss, smooth_loss, lambd1, lambd2):
        super(HybridLoss, self).__init__()
        self.umap_loss = umap_loss
        self.recon_loss = recon_loss
        self.smooth_loss = smooth_loss
        self.lambd1 = lambd1
        self.lambd2 = lambd2

    def forward(self, edge_to, edge_from, a_to, a_from, embeded_to, coeff, outputs):
        embedding_to, embedding_from = outputs["umap"]
        recon_to, recon_from = outputs["recon"]

        recon_l = self.recon_loss(edge_to, edge_from, recon_to, recon_from, a_to, a_from)
        umap_l = self.umap_loss(embedding_to, embedding_from)
        smooth_l = self.smooth_loss(embedding_to, embeded_to, coeff)

        loss = umap_l + self.lambd1 * recon_l + self.lambd2 * smooth_l

        return umap_l, recon_l, smooth_l, loss

