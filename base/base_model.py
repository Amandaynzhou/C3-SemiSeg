import logging
import numpy as np
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_names = []

    def forward(self):
        raise NotImplementedError

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(f'Nbr of trainable parameters: {nbr_params}')

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + f'\nNbr of trainable parameters: {nbr_params}'
        # return summary(self, input_shape=(2, 3, 224, 224))

    def initialize(self, args, rank, local_rank, ):
        self.local_rank = local_rank
        self.rank = rank
        self.dist = args.distributed
        self.args = args
        self.gpu_ids = args.gpu_ids if not args.distributed else [args.gpu]
        if self.dist:
            self.device = self.local_rank
        else:
            self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
