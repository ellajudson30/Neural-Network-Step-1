import torch
import torch.nn as nn
from rk45utils import N_FEATURES

# Network used for stage 1 and the seed ensemble that averages several trained copies
#------------------------------------------------------------------------

# Seeds of the networks in the ensemble (one cross_valid.py run per seed, see
# train_ensemble.py). Networks trained with different seeds disagree a lot when they
# drive the solver; averaging them removes much of that variation.
ENSEMBLE_SEEDS = [0, 1, 2]

def weights_path(seed):
    return f'model_weights_s1_seed{seed}.pth'

class NN3(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,32),
            nn.Tanh(),
            nn.Linear(32,1)
        )

    def forward(self, x):
        output = self.net(x)
        return output

class Ensemble(nn.Module):
    """Averages the members' predicted log ratios (a geometric mean of their ratios)."""
    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x):
        return torch.stack([m(x) for m in self.members]).mean(dim=0)

def load_ensemble(seeds=ENSEMBLE_SEEDS):
    members = []
    for seed in seeds:
        model = NN3()
        model.load_state_dict(torch.load(weights_path(seed)))
        model.eval()
        members.append(model)
    return Ensemble(members).eval()
