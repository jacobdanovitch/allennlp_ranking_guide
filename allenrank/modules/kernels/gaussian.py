from typing import Dict, Iterator, List

import torch
import torch.nn as nn

from allennlp.modules.matrix_attention import CosineMatrixAttention
from allennlp.data import TextFieldTensors

from allenrank.modules.kernels import KernelFunction

@KernelFunction.register('gaussian')
class GaussianKernel(KernelFunction):
    def __init__(
        self,
        n_kernels: int,
    ):
        super().__init__()
        
        # static - kernel size & magnitude variables
        self.register_buffer('mu', self.kernel_mus(n_kernels).view(1, 1, 1, n_kernels))
        self.register_buffer('sigma', self.kernel_sigmas(n_kernels).view(1, 1, 1, n_kernels))

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()
        
    def forward(
        self, 
        query_embeddings: TextFieldTensors, 
        candidates_embeddings: TextFieldTensors,
        query_mask: torch.Tensor = None,
        candidates_mask: torch.Tensor = None
    ):
        mask = query_mask.unsqueeze(-1).float() @ candidates_mask.unsqueeze(-1).transpose(-1, -2).float()
        
        #
        # cosine matrix
        # -------------------------------------------------------
        
        # shape: (batch, query_max, doc_max)
        cosine_matrix = (self.cosine_module(query_embeddings, candidates_embeddings) * mask).unsqueeze(-1)
        
        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------
        
        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * mask.unsqueeze(-1)
        
        per_kernel_query = torch.sum(kernel_results_masked, 2)
        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) * 0.01
        log_per_kernel_query_masked = log_per_kernel_query * query_mask.unsqueeze(-1) # make sure we mask out padding values

        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 
        return per_kernel
    
    
    def cuda(self, device=None):
        self = super().cuda(device)
        self.mu = self.mu.cuda(device)
        self.sigma = self.sigma.cuda(device)
        return self 


    def kernel_mus(self, n_kernels: int):
        """
        get the mu for each guassian kernel. Mu is the middle of each bin
        :param n_kernels: number of kernels (including exact match). first one is exact match
        :return: l_mu, a list of mu.
        """

        bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
        
        exact_match = torch.tensor([1.])
        mu_bins = torch.linspace(1 - bin_size/2, -(1 - bin_size/2), n_kernels-1)
        
        l_mu = torch.cat([exact_match, mu_bins], dim=0)
        return nn.Parameter(l_mu, requires_grad=False)

    def kernel_sigmas(self, n_kernels: int):
        """
        get sigmas for each guassian kernel.
        :param n_kernels: number of kernels (including exactmath.)
        :return: l_sigma, a list of simga
        """
        bin_size = 2.0 / (n_kernels - 1)
        exact_match = torch.tensor([0.0001])  # for exact match. small variance -> exact match
        sigma_windows = (0.5 * bin_size) * torch.ones(n_kernels - 1)
        
        l_sigma = torch.cat([exact_match, sigma_windows], dim=0)
        return l_sigma
    
    
    
def make_parameter(
    tensor: torch.Tensor
):
    return nn.Parameter(torch.tensor(tensor), requires_grad=False)