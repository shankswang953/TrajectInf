import sys
import os

from pkg_resources import normalize_path
from Args import get_notebook_args, Args
sys.path.insert(0, os.path.abspath('../..'))
from src.DataLoad import load_source_data
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from src.Epoch import train_epoch
from src.utility import precompute_gaussian_params
from src.Neural import CNF, RunningAverageMeter
from src.plot import *
from src.PreProcess import *
from src.DataLoad import create_mixture_gaussian, normalize_to_unit_cube

args = Args()
print(f"Default hidden dimension: {args.time_steps}")
print(f"Default ot dimension: {args.long_time_steps}")
print(f"Default data path: {args.data_path}")
print(f"Default time labels: {args.time_labels}")
print(f"Default ot dimension: {args.otdim}")
print(f"Default time points: {args.time_points}")

data_train= load_source_data(args.data_path, args.time_labels, args.support_points)

import matplotlib.pyplot as plt


device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    
writer = SummaryWriter(log_dir=os.path.join(args.results_dir, 'tensorboard'))

primary_data = [data.to(device) for data in data_train] 
# record the number of samples in each group
lengths = [data.shape[0] for data in primary_data]

# concatenate and normaliz

total_primal = torch.cat(primary_data, dim=0)


total_primal, norm_params = normalize_to_unit_cube(total_primal)

# split back according to the recorded lengths
split_points = torch.cumsum(torch.tensor(lengths), dim=0).tolist()
start = 0
normalized_data = []

for end in split_points:
    normalized_data.append(total_primal[start:end])
    start = end

# update primary_data to the normalized data
primary_data = normalized_data 

manifold = torch.cat(primary_data, dim=0)
manifold_mixture = create_mixture_gaussian(manifold, args.sigma, args.adaptive_sigma, device)

density_precompute_list = []
for i in range(len(args.time_points)):
    z_curr_primal = primary_data[i]
    sigma = args.sigma
    adaptive_sigma = args.adaptive_sigma
    z_curr_mixture = create_mixture_gaussian(z_curr_primal, sigma, adaptive_sigma, device)
    weights, means, precisions = precompute_gaussian_params(z_curr_primal, sigma, device)
    density_precompute_list.append((weights, means, precisions, z_curr_mixture))
    
time_steps_list = []
for t_idx in range(len(args.time_points) - 1):
    dt = args.dt
    t0, t1 = args.time_points[t_idx], args.time_points[t_idx + 1]
    
    time_steps_short_backward = torch.linspace(t1, t0, int((t1-t0)/dt)+1).to(device)
    time_steps_short_forward = torch.linspace(t0, t1, int((t1-t0)/dt)+1).to(device)
    
    t_init = args.time_points[0]
    time_steps_long_backward = torch.linspace(t1, t_init, int((t1-t_init)/dt)+1).to(device)
    time_steps_long_forward = torch.linspace(t_init, t1, int((t1-t_init)/dt)+1).to(device)
    
    time_steps_list.append((time_steps_short_backward, time_steps_short_forward, 
                            time_steps_long_backward, time_steps_long_forward))
print(time_steps_list)

func = CNF(in_out_dim=args.otdim, hidden_dim=args.hidden_dim, n_hiddens=args.n_hiddens,activation=args.activation, UnbalancedOT=args.UnbalancedOT).to(device)
#optimizer = optim.Adam(func.parameters(), lr=args.lr)
optimizer = optim.AdamW(
func.parameters(),
lr=args.lr,
weight_decay=0.001
)

total_steps = args.niters
gamma = (2e-5 / args.lr) ** (2.0 / total_steps) 

loss_meter = RunningAverageMeter()

scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma=gamma,
    verbose=True
)


secondary_space= load_source_data(args.sync_data_path, args.time_labels, args.support_points)
    
secondary_data = [data.to(device) for data in secondary_space] 
# record the number of samples in each group
lengths = [data.shape[0] for data in secondary_data]

# concatenate and normaliz

total_secondary = torch.cat(secondary_data, dim=0)



total_secondary, norm_params = normalize_to_unit_cube(total_secondary)

# split back according to the recorded lengths
split_points = torch.cumsum(torch.tensor(lengths), dim=0).tolist()
start = 0
normalized_secondary = []

for end in split_points:
    normalized_secondary.append(total_secondary[start:end])
    start = end

# update secondary_data to the normalized data
secondary_data = normalized_secondary 
secondary_mixture = create_mixture_gaussian(total_secondary, args.sigma, args.adaptive_sigma, device)



train_results = train_epoch(
    func=func,
    primary_data=primary_data,
    time_points=args.time_points,
    time_steps_list=time_steps_list,
    density_precompute_list=density_precompute_list,
    optimizer=optimizer,
    device=device,
    total_primal=total_primal,
    secondary_data=secondary_data,
    manifold_mixture=manifold_mixture,
    secondary_mixture=secondary_mixture,
    args=args,
    writer=writer,
    loss_meter=loss_meter,
    scheduler=scheduler,
    epoch=0
)