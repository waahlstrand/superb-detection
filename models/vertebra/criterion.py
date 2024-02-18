from models.backbones.realnvp.models import RealNVP
from torch import Tensor
import torch
import torch.nn as nn
import math

from typing import *


class RLELoss(nn.Module):

    def __init__(self, n_keypoints: int = 6, n_dims: int = 2, prior: Literal["laplace", "gaussian"] = "laplace") -> None:
        
        super().__init__()
        self.eps = 1e-9
        self.flow = RealNVP()
        self.prior = prior
        self.n_keypoints = n_keypoints
        self.n_dims = n_dims

        self.log_phi = torch.vmap(self.flow.log_prob, in_dims=1, out_dims=1, chunk_size=8)

    def forward(self, mu: Tensor, sigma: Tensor, x: Tensor) -> Tensor:
        """
        Args:
            mu (Tensor): (B, K x 2) The predicted mean of the distribution
            sigma (Tensor): (B, K x 2) The predicted standard deviation of the distribution
            x (Tensor): (N, 2) The query point, or the ground truth point in the case of training

        Returns:
            Tensor: The log-likelihood of the query point under the distribution
        """
        # B, KD = mu.shape
        # D = 2
        # K = KD // D
        # x = x.reshape(-1, D)
        # N = x.shape[0] 



        mu, sigma, x = mu.reshape(-1, self.n_keypoints, self.n_dims), sigma.reshape(-1, self.n_keypoints, self.n_dims), x.reshape(-1, self.n_keypoints, self.n_dims)

        # Calculate the deviation from a sample x
        error = (mu - x) / (sigma + self.eps) # (B x K, N, D)
        log_phi = self.flow.log_prob(error.view(-1, self.n_dims)).view(-1, self.n_keypoints, 1)
        log_sigma = torch.log(sigma).view(-1, self.n_keypoints, self.n_dims)

        match self.prior:
            case "laplace":
                log_q = torch.log(2 * sigma) + torch.abs(error)

            case "gaussian":
                log_q = torch.log(sigma * math.sqrt(2 * math.pi)) + 0.5 * error**2

            case _:
                raise NotImplementedError("Prior not implemented")
            
        nll = log_sigma - log_phi + log_q
        nll /= len(nll)
        nll = nll.sum()

        return nll

        # Calculate the log-likelihood from the flow
        # log_phi = self.log_phi(error).view(-1, len(x), 1).repeat(1, 1, D) # (B x K, N, D)
        # log_phi = self.flow.log_prob(error.view(-1, 2))#.view(-1, N, 1).repeat(1, 1, D)
        
        # log_sigma = torch.log(sigma)#.repeat(1, len(x), 1) # (B x K, N, D)

        # match self.prior:
        #     case "laplace":
        #         log_q = torch.log(2 * sigma) + torch.abs(error) # (B x K, N, D)
        #     case "gaussian":
        #         log_q = torch.log(sigma * math.sqrt(2 * math.pi)) + 0.5 * error**2
        #     case _:
        #         raise NotImplementedError("Prior not implemented")

        # nll = log_sigma.mean() - log_phi.mean() + log_q.mean() # (B x K, N, D)

        # # nll = nll.view(B*N, K, D) # (B x N, K)
        # return nll


        # BN, KD = mu.shape
        # D = 2
        # N = x.shape[0]
        # mu = mu.reshape(-1, 1, x.shape[1])
        # sigma = sigma.reshape(-1, 1, x.shape[1])
        # # Calculate the deviation from a sample x
        # error = (mu - x) / (sigma + self.eps) # (B*N, K*D)

        # # (B*N, K*D)
        # log_phi = self.flow.log_prob(error.view(-1, D)).view(BN, -1, 1)

        # # (B*N, K*D)

        # if self.prior == "gaussian":
        #     log_q = torch.log(sigma * math.sqrt(2 * math.pi)) + 0.5 * error**2
        # elif self.prior == "laplace":
        #     log_q = torch.log(sigma * 2) + torch.abs(error)

        # print(log_q.shape, log_phi.shape, sigma.shape)
        # log_sigma = torch.log(sigma).view(-1, 1, 2).sum(-1, keepdim=True).repeat(KD//D, 1, 1)
        # log_q     = log_q.view(-1, 1, 2).sum(-1, keepdim=True)

        # # sigma = sigma.view(BN, KD//D, D, )
        # # log_q = log_q.view(BN, KD//D, D, )

        # # (B*N, ) by broadcasting (possibly incorrect)
        # loss = log_sigma - log_phi + log_q

        # return loss.mean()

        # return nll
    