import torch
import torch_geometric
from torch import Tensor
import torch.nn.functional as F
from typing import Callable, Optional, Tuple, Union
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros, reset
from torch_geometric.nn import DeepGCNLayer
from torch_geometric.nn.conv import ResGatedGraphConv
from torch_geometric.typing import Adj, OptTensor, PairTensor, Tensor, Size

# ================================================================================================
# Custom graph convolution layer
# ================================================================================================

class CustomConv(MessagePassing):

    def __init__(self, feats_dim: int, out_dim: int, edge_mlp: Callable, 
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.edge_mlp = edge_mlp
        self.node_mlp1 = torch.nn.Sequential(
            torch.nn.Linear(2*feats_dim,feats_dim, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Linear(feats_dim,out_dim))
        self.node_mlp2 = torch.nn.Sequential(
            torch.nn.Linear(feats_dim,feats_dim, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Linear(feats_dim,out_dim))

        self.reset_parameters()

    def reset_parameters(self):

        for layer in self.edge_mlp:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.node_mlp1:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.node_mlp2:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: Tensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x = (x, x)

        a = x[0]
        b = x[1]

        # propagate_type: (a: Tensor, b: Tensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, a=a, b=b, edge_attr=edge_attr, size=size)
        return out + self.node_mlp2(x[1])

    def message(self, a_j: Tensor, b_i: Tensor, edge_attr: Tensor) -> Tensor:
        out = self.edge_mlp(edge_attr) * self.node_mlp1(torch.cat([a_j,b_i], dim=-1))
        return out

# ================================================================================================
# Autoencoder class 
# ================================================================================================

class GAE(torch.nn.Module):
    """ General Graph Autoencoder 
    Args:
        encoder (Module): The encoder module.
        decoder (Module): The decoder module
    """
    def __init__(self, encoder, decoder):
        super(GAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        GAE.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def encode(self, *args, **kwargs):
        """Runs the encoder"""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Runs the decoder and computes edge values"""
        return self.decoder(*args, **kwargs)

    def forward(self, x, edge_index, edge_attr, mask):
        z = self.encode(x, edge_index, edge_attr)
        return self.decoder(z, mask)

# ================================================================================================
# Autoencoder + GRU class 
# ================================================================================================

class GAEGRU(torch.nn.Module):
    """ General Graph Autoencoder + GRU
    Args:
        encoder (Module): The encoder module.
        decoder (Module): The decoder module
    """
    def __init__(self, encoder, decoder):
        super(GAEGRU, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        GAEGRU.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def encode(self, *args, **kwargs):
        """Runs the encoder"""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Runs the decoder and computes edge values"""
        return self.decoder(*args, **kwargs)

    def forward(self, x, edge_index, edge_attr, mask, h=None):
        z = self.encode(x, edge_index, edge_attr, h)
        return z, self.decoder(z, mask)


# ==================================================================================================
# Encoders
# ==================================================================================================

class CustomGraphConv(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, latent_dim, edge_dim, n_layers, dropout, batch):
        super(CustomGraphConv, self).__init__()
        self.lin1 = torch.nn.Linear(input_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, latent_dim)
        self.n_layers = n_layers
        self.batch = batch

        self.att = torch.nn.ModuleList([])
        self.layer = torch.nn.ModuleList([])
        for n in range(n_layers):
            edge_mlp = torch.nn.Sequential(
                torch.nn.Linear(edge_dim, hidden_dim), torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim))
            norm = torch.nn.LayerNorm(hidden_dim, elementwise_affine=True)
            conv = CustomConv(hidden_dim, hidden_dim, edge_mlp, aggr='max')
            act = torch.nn.ReLU(inplace=True)
            self.layer.append(DeepGCNLayer(conv, norm, act, block='res+', dropout=dropout))
            self.att.append(ResGatedGraphConv(hidden_dim, hidden_dim, aggr='mean'))

    def forward(self, x, edge_index, edge_attr):

        # First layer
        z = self.lin1(x)
        z = self.layer[0](z, edge_index, edge_attr)
        # Intermediate layers
        for n in range(self.n_layers-1):
            z = self.att[n](z,edge_index)
            z = self.layer[n+1](z,edge_index,edge_attr)
        return self.lin2(z)

# ==================================================================================================
# Decoders
# ==================================================================================================

class InnerProductDecoder(torch.nn.Module):

    def __init__(self, batch_size):
        super(InnerProductDecoder, self).__init__()
        self.batch_size = batch_size
    
    def forward(self, z, mask):
        value = torch.mm(z, z.t())
        value[mask < 1e-8] = -1e3
        # Split to batch dimension and apply softmax to each batch
        return torch.nn.Softmax(dim=1)(value.view(self.batch_size,-1)).view_as(mask)

# ==================================================================================================
# Gated Recurrent Unit
# ==================================================================================================

class GraphConvGRU(torch.nn.Module):
    """ Gated Recurrent Unit Cell on top of a GNN model"""

    def __init__(self, input_dim, hidden_dim, latent_dim, edge_dim, n_layers, dropout, batch):
        super(GraphConvGRU, self).__init__()

        self.conv = CustomGraphConv(input_dim, hidden_dim, latent_dim, edge_dim, n_layers, dropout, batch)
        self.lin_u = torch.nn.Linear(2*latent_dim, latent_dim)#, bias=False) 
        self.lin_r = torch.nn.Linear(2*latent_dim, latent_dim)#, bias=False) 
        self.lin_h = torch.nn.Linear(2*latent_dim, latent_dim)#, bias=False) 
        self.latent_dim = latent_dim

    def set_hidden_state(self, x, h):
        if h is None:
            h = torch.zeros(x.shape[0], self.latent_dim).to(x.device) 
        return h

    def calculate_update_gate(self, x, edge_index, edge_attr, h):
        u = torch.cat([self.conv(x, edge_index, edge_attr), h], axis=1) 
        u = self.lin_u(u)
        return torch.sigmoid(u)

    def calculate_reset_gate(self, x, edge_index, edge_attr, h):
        r = torch.cat([self.conv(x, edge_index, edge_attr), h], axis=1)
        r = self.lin_r(r) 
        return torch.sigmoid(r)

    def calculate_candidate_state(self, x, edge_index, edge_attr, h, r):
        h_c = torch.cat([self.conv(x, edge_index, edge_attr), h * r], axis=1)  
        h_c = self.lin_h(h_c)
        return torch.relu(h_c)

    def calculate_hidden_state(self, z, h, h_c):
        return z*h + (1-z)*h_c 

    def forward(self, x, edge_index, edge_attr, h=None):
        h = self.set_hidden_state(x, h)
        z = self.calculate_update_gate(x, edge_index, edge_attr, h)
        r = self.calculate_reset_gate(x, edge_index, edge_attr, h)
        h_c = self.calculate_candidate_state(x, edge_index, edge_attr, h, r)
        h = self.calculate_hidden_state(z, h, h_c) 
        return h

# ================================================================================================
# Loss functions
# ================================================================================================

def BCELoss(A_pred, A_data, pad_mask):
    """
    Binary cross entropy-based loss function
    """
    return F.binary_cross_entropy(A_pred, A_data, size_average=False, weight=pad_mask)
