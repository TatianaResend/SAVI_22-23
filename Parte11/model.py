#!/usr/bin/env python3
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = torch.nn.Linear(1,1)

    def forward(self,xs):
        ys = self.layer1(xs)

        return  ys