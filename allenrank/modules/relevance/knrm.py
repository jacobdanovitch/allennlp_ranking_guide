from typing import Dict, Iterator, List

import torch
import torch.nn as nn

from allennlp.modules.matrix_attention import CosineMatrixAttention
from allennlp.data import TextFieldTensors

from allenrank.modules.relevance.base import RelevanceMatcher

import torchsnooper

@RelevanceMatcher.register('knrm')
class KNRM(RelevanceMatcher):
    '''
    Paper: End-to-End Neural Ad-hoc Ranking with Kernel Pooling, Xiong et al., SIGIR'17

    Adapted from: https://github.com/sebastian-hofstaetter/transformer-kernel-ranking/blob/master/matchmaker/models/knrm.py
    '''
    def __init__(
        self,
        n_kernels: int,
        **kwargs
    ):
        kwargs['input_dim'] = n_kernels
        super().__init__(**kwargs)

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
        candidates_mask: torch.Tensor = None,
        output_secondary_output: bool = False
    ) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        query_by_doc_mask = torch.bmm(query_mask.unsqueeze(-1).float(), candidates_mask.unsqueeze(-1).transpose(-1, -2).float())
        query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)

        #
        # cosine matrix
        # -------------------------------------------------------

        # shape: (batch, query_max, doc_max)
        cosine_matrix = self.cosine_module(query_embeddings, candidates_embeddings)
        cosine_matrix_masked = cosine_matrix * query_by_doc_mask
        cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)

        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------
        
        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * query_by_doc_mask_view

        per_kernel_query = torch.sum(kernel_results_masked, 2)
        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) * 0.01
        log_per_kernel_query_masked = log_per_kernel_query * query_mask.unsqueeze(-1) # make sure we mask out padding values

        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

        ##
        ## "Learning to rank" layer - connects kernels with learned weights
        ## -------------------------------------------------------

        dense_out = self.dense(per_kernel)
        score = torch.squeeze(dense_out,1)

        if output_secondary_output:
            query_mean_vector = query_embeddings.sum(dim=1) / query_mask.sum(dim=1).unsqueeze(-1)
            return score, {"score":score,"per_kernel":per_kernel,"query_mean_vector":query_mean_vector,"cosine_matrix_masked":cosine_matrix_masked}
        return score
    
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
        l_mu = [1.0]
        if n_kernels == 1:
            return l_mu

        bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
        l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
        for i in range(1, n_kernels - 1):
            l_mu.append(l_mu[i] - bin_size)
        return nn.Parameter(torch.tensor(l_mu, device=self.dense.weight.device), requires_grad=False)

    def kernel_sigmas(self, n_kernels: int):
        """
        get sigmas for each guassian kernel.
        :param n_kernels: number of kernels (including exactmath.)
        :param lamb:
        :param use_exact:
        :return: l_sigma, a list of simga
        """
        bin_size = 2.0 / (n_kernels - 1)
        l_sigma = [0.0001]  # for exact match. small variance -> exact match
        if n_kernels == 1:
            return l_sigma

        l_sigma += [0.5 * bin_size] * (n_kernels - 1)
        return nn.Parameter(torch.tensor(l_sigma, device=self.dense.weight.device), requires_grad=False)