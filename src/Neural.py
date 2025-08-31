import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torchdiffeq import odeint

class CNF(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, n_hiddens, activation, UnbalancedOT):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.n_hiddens = n_hiddens
        self.activation = activation
        self.width = hidden_dim * 2

        self.UnbalancedOT = UnbalancedOT
        if UnbalancedOT:
            self.hyper_net1 = HyperNetwork1(in_out_dim, hidden_dim, n_hiddens,activation) #v= dx/dt
            self.hyper_net2 = HyperNetwork2(in_out_dim, hidden_dim, activation) #g
        else:
            self.balanced_net = BalancedNetwork(in_out_dim, hidden_dim, width=self.width)

    def forward(self, t, states):
        z = states[0]
        logp_z = states[1]
        batchsize = z.shape[0]

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
                    
            if self.UnbalancedOT:
                dz_dt = self.hyper_net1(t, z)  
                g = self.hyper_net2(t, z)
                dlogp_z_dt = g - trace_df_dz(dz_dt, z).view(batchsize, 1)
                
                return (dz_dt, dlogp_z_dt, g)
            else:
                W, B, U = self.balanced_net(t)

                Z = torch.unsqueeze(z, 0).repeat(self.width, 1, 1)
                h = torch.tanh(torch.matmul(Z, W) + B)
                dz_dt = torch.matmul(h, U).mean(0)           
                #dz_dt = self.balanced_net(t, z)
                dlogp_z_dt = -trace_df_dz(dz_dt, z).view(batchsize, 1)
                
                return (dz_dt, dlogp_z_dt)


def trace_df_dz(f, z):

    sum_diag = 0.0
    for i in range(z.shape[1]):
     
        sum_diag += torch.autograd.grad(f[:, i].sum(),
                                        z, 
                                        create_graph=True)[0].contiguous()[:, i].contiguous()
        
    return sum_diag.contiguous()

class BalancedNetwork(nn.Module):

    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()

        blocksize = width * in_out_dim

        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3 * blocksize + width)

        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.blocksize = blocksize
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, t):
        
        params = t.reshape(1, 1)
        params = self.leaky_relu(self.fc1(params))
        params = self.leaky_relu(self.fc2(params))
        params = self.fc3(params)

        params = params.reshape(-1)
        W = params[:self.blocksize].reshape(self.width, self.in_out_dim, 1)

        U = params[self.blocksize:2 * self.blocksize].reshape(self.width, 1, self.in_out_dim)

        G = params[2 * self.blocksize:3 * self.blocksize].reshape(self.width, 1, self.in_out_dim)
        U = U * torch.sigmoid(G)

        B = params[3 * self.blocksize:].reshape(self.width, 1, 1)
        
        return [W, B, U]
    
    
class HyperNetwork1(nn.Module):
    # input x, t to get v= dx/dt
    def __init__(self, in_out_dim, hidden_dim, n_hiddens, activation='Tanh'):
        super().__init__()
        Layers = [in_out_dim+1]
        for i in range(n_hiddens):
            Layers.append(hidden_dim)
        Layers.append(in_out_dim)
        
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()     

        self.net = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(Layers[i], Layers[i + 1]),
                self.activation,
            )
                for i in range(len(Layers) - 2)
            ]
        )
        self.out = nn.Linear(Layers[-2], Layers[-1])

    def forward(self, t, x):
        # x is N*2
        batchsize = x.shape[0]   
        t = t.clone().detach().to(x.device).repeat(batchsize).reshape(-1, 1)                    
        t.requires_grad=True
        state  = torch.cat((t,x),dim=1)
        
        ii = 0
        for layer in self.net:
            if ii == 0:
                x = layer(state)
            else:
                x = layer(x)
            ii =ii+1
        x = self.out(x)
        return x


class HyperNetwork2(nn.Module):
    # input x, t to get g
    def __init__(self, in_out_dim, hidden_dim, activation='Tanh'):
        super().__init__()
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        self.net = nn.Sequential(
            nn.Linear(in_out_dim+1, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,1))
        
    def forward(self, t, x):
        # x is N*2
        batchsize = x.shape[0]   
        t = t.clone().detach().to(x.device).repeat(batchsize).reshape(-1, 1)            
        t.requires_grad=True
        state  = torch.cat((t,x),dim=1)
        return self.net(state)
    
    

class RunningAverageMeter(object):

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
        
def ggrowth(t,y,func,device):
    y_0 = torch.zeros(y[0].shape).type(torch.float32).to(device)
    y_00 = torch.zeros(y[1].shape).type(torch.float32).to(device)                       
    gg = func.forward(t, y)[2]
    return (y_0,y_00,gg)


class SIRENLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=10):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            bound = 1 / self.linear.in_features
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class SIRENNetwork(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, n_layers=5, omega_0=10):
        super().__init__()
        layers = [SIRENLayer(in_out_dim + 1, hidden_dim, omega_0)]
        for _ in range(n_layers - 2):
            layers.append(SIRENLayer(hidden_dim, hidden_dim, omega_0))
        layers.append(nn.Linear(hidden_dim, in_out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t, z):
        batch_size = z.shape[0]
        t_expanded = t.expand(batch_size, 1)
        input_cat = torch.cat([z, t_expanded], dim=1)
        return self.net(input_cat)