import matplotlib.pyplot as plt
import numpy as np
import torch
from model import get_model

qs, qEs = np.load("qs-0.npz").values()
ps, pEs = np.load("ps.npz").values()

model = get_model()

model.load_state_dict(torch.load("emd-kr-toy-0.pt"))

plt.scatter(ps[:, 0], ps[:, 1], s=pEs * 1000, c="crimson")
plt.scatter(qs[:, 0], qs[:, 1], s=qEs * 1000, c="royalblue")

nb = 100

minx = min(qs[:, 0].min(), ps[:, 0].min())
maxx = max(qs[:, 0].max(), ps[:, 0].max())
miny = min(qs[:, 1].min(), ps[:, 1].min())
maxy = max(qs[:, 1].max(), ps[:, 1].max())

X, Y = np.meshgrid(np.linspace(minx, maxx, nb), np.linspace(miny, maxy, nb))

XY = np.vstack([X.flatten(), Y.flatten()]).T
with torch.no_grad():
    Z = model(torch.tensor(XY).float())

plt.pcolormesh(X, Y, Z.numpy().reshape(nb, nb), shading="auto", alpha=0.5)
plt.colorbar()
# plt.scatter(qbins[:, 0], qbins[:, 1], c=q, s=q*100000, alpha=.5)
# plt.scatter(pbins[:, 0], pbins[:, 1], c=p, s=p*100000, alpha=.5, marker='x')
plt.show()
