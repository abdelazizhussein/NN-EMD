import torch
from monotonenorm import direct_norm, GroupSort
from mup import MuReadout, make_base_shapes, set_base_shapes, MuSGD, MuAdam


class TimesN(torch.nn.Module):
    def __init__(self, n:float):
        super().__init__()
        self.n = n
    def forward(self, x):
        return self.n*x

# def init_weights(m):
#     if isinstance(m, torch.nn.Linear):
#         torch.nn.init.orthogonal_(m.weight)

def get_model(width,dev=None):
    model = torch.nn.Sequential(
        direct_norm(torch.nn.Linear(2, width), kind="two-inf", always_norm=False),
        #torch.nn.Linear(2, width)
        GroupSort(2),
        direct_norm(torch.nn.Linear(width, width), kind="inf", always_norm=False),
        GroupSort(2),
        direct_norm(torch.nn.Linear(width, width), kind="inf", always_norm=False),
        GroupSort(2),
        direct_norm(torch.nn.Linear(width, width), kind="inf", always_norm=False),
        GroupSort(2),
        direct_norm(torch.nn.Linear(width, width), kind="inf", always_norm=False),
        GroupSort(2),
        direct_norm(MuReadout(width, 1), kind="inf", always_norm=False),
        TimesN(1)
    ).to(dev or "cpu")
    #model.apply(init_weights)
    return model

