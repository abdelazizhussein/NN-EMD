import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import numpy as np
import ot
from tqdm import tqdm
from copy import deepcopy
from model import get_model
import matplotlib.pyplot as plt
from IPython import embed
from mup import MuReadout, make_base_shapes, set_base_shapes, MuSGD, MuAdam,init

torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False

torch.random.manual_seed(5)
np.random.seed(10)

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = get_model(width = 500,dev = dev)

m1 = get_model(width=500)

# for name, module in m1.named_modules():
#     if isinstance(module, torch.nn.Linear):
#         print(module.weight.infshape[0])

m2 = get_model(width=500)

set_base_shapes(model, m1, delta=m2)

### Replace your custom init, if any
for param in model.parameters():
    ### If initializing manually with fixed std or bounds,
    ### then replace with same function from mup.init
    torch.nn.init.uniform_(param, -1, 1)
    #init.uniform_(param, -1, 1)

dim = 2
size = 10000
n = 20
n_targets = 10

dE = 0

n_tracks_p = 2
n_tracks_q = 1

ps = np.random.randn(n_tracks_p, 2)
#ps =np.array([[0,1]])
pEs = np.random.rand(n_tracks_p)
#pEs = np.array([1])
pEs = pEs / pEs.sum()
np.savez(f"ps.npz", ps=ps, pEs=pEs)

for i in range(1):
    qs = np.random.randn(n_tracks_q, 2)
    #qs = np.array([[1,0]])
    qEs = np.random.rand(n_tracks_q)
    #qEs = np.array([5])
    qEs = qEs / qEs.sum() * (1 + dE)

    np.savez(f"qs-{i}.npz", qs=qs, qEs=qEs)
    pEs_cheat = np.concatenate((pEs, np.array([dE])))

    M = ot.dist(ps, qs, metric="euclidean")
    M = np.vstack((M, np.ones(M.shape[1])))
    distance = ot.emd2(pEs_cheat, qEs, M)

    EPOCHS = 60000
    lr_init = 2e-2
    lr_final = 1e-4
    gamma = (lr_final / lr_init) ** (1 / EPOCHS)

    pst = torch.from_numpy(ps).float().to(dev)
    qst = torch.from_numpy(qs).float().to(dev)
    pEst = torch.from_numpy(pEs).float().to(dev).view(-1, 1)
    qEst = torch.from_numpy(qEs).float().to(dev).view(-1, 1)

    max_emd = 0
    #optim = torch.optim.Adam(model.parameters(), lr=lr_init)
    optim = MuAdam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=gamma)

    bar = tqdm(range(EPOCHS))
    loss_plt = []
    for epoch in bar:
        mp = model(pst)
        mq = model(qst)
        E0 = (mp * pEst).sum()
        E1 = (mq * qEst).sum()
        emd = E1 - E0 - dE * (mq.max() - 1)
        if emd.item() > max_emd:
            sd = deepcopy(model.state_dict())
            max_emd = emd.item()
            delta = (max_emd - distance) / distance * 100
            loss_plt.append(delta/100)
            message = f"{max_emd:.3f} vs {distance:.3f} - Delta: {delta:.2f}% @ {epoch}"
            bar.set_description(message)
        # update model
        loss = -emd
        loss.backward()
        optim.step()
        scheduler.step()
        optim.zero_grad()
    
    plt.figure()
    x = np.linspace(0,EPOCHS,range(loss_plt)-1)
    plt.plot(x,np.abs(loss_plt)*100)
    plt.ylim(0,15)
    plt.show()

    
    torch.save(sd, f"emd-kr-toy-{i}.pt")

