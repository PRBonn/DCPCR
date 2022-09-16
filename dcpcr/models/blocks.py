# from pykeops.torch import Vi, Vj
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_
from torch.nn.parameter import Parameter


class STNkd(nn.Module):
    def __init__(self, k=64, norm=True):
        super(STNkd, self).__init__()
        self.conv1 = nn.Linear(k, 64)
        self.conv2 = nn.Linear(64, 128)
        self.conv3 = nn.Linear(128, 1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        # exchanged Batchnorm1d by Layernorm
        self.bn1 = nn.LayerNorm(64) if norm else nn.Identity()
        self.bn2 = nn.LayerNorm(128) if norm else nn.Identity()
        self.bn3 = nn.LayerNorm(1024) if norm else nn.Identity()
        self.bn4 = nn.LayerNorm(512) if norm else nn.Identity()
        self.bn5 = nn.LayerNorm(256) if norm else nn.Identity()

        self.k = k

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, -2, keepdim=True)[0]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.k, device=x.device, dtype=x.dtype)
        shape = x.shape[:-1]+(1,)
        iden = iden.repeat(*shape)
        x = x.view(iden.shape) + iden
        return x


class PointNetFeat(nn.Module):
    def __init__(self, in_dim=3, out_dim=1024, input_transform=True, feature_transform=False, norm=True):
        super(PointNetFeat, self).__init__()
        self.conv1 = nn.Linear(in_dim, 64)
        self.conv2 = nn.Linear(64, 128)
        self.conv3 = nn.Linear(128, out_dim)
        self.bn1 = nn.LayerNorm(64) if norm else nn.Identity()
        self.bn2 = nn.LayerNorm(128) if norm else nn.Identity()
        self.bn3 = nn.LayerNorm(out_dim) if norm else nn.Identity()
        self.feature_transform = feature_transform
        self.input_transform = input_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64, norm=norm)
        if self.input_transform:
            self.stn = STNkd(k=in_dim, norm=norm)

    def forward(self, x):
        if self.input_transform:
            trans = self.stn(x)
            x = torch.matmul(x, trans)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = torch.matmul(x, trans_feat)
        else:
            trans_feat = None

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return x


def softmax(t, mask=None, dim=0, epsilon=1e-9):
    if mask is not None:
        t_exp = torch.exp(t)*mask.float()
        # print(t.max())
        sm = t_exp/(torch.sum(t_exp, dim=dim, keepdim=True)+epsilon)
        return sm
    else:
        return F.softmax(t, dim)


def norm(t: torch.Tensor, mask=None, tau: int = 3):
    t = softmax(t, mask, dim=-1)
    if tau != 1:
        t = t**tau / (t**tau).sum(dim=-1, keepdim=True)
    return t


class Attention(nn.Module):
    def __init__(self, tau, attention_normalization='softmax'):
        super().__init__()
        self.tau = tau

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, k_mask=None, q_mask=None):
        d_k_sq = q.shape[-1]**0.5
        w = torch.matmul(q, k.transpose(-1, -2))/d_k_sq
        mask = None
        if k_mask is not None and q_mask is not None:
            mask = torch.min(k_mask.transpose(-1, -2).expand(
                w.shape), q_mask.expand(w.shape))

        # To prevent for stuff getting 'inf'
        w = w-w.max(axis=-1, keepdim=True)[0]
        w = norm(w, mask=mask, tau=self.tau)
        f = torch.matmul(w, v)
        return f, w


def weightedNanMean(tensor, weights=None, axis=None, keepdims=False):
    if weights is None:
        return(torch.nanmean(tensor, axis=axis, keepdims=keepdims))
    else:
        return torch.sum(tensor*weights, axis=axis, keepdims=keepdims)/(torch.sum(weights, axis=axis, keepdims=keepdims)+1e-8)


class SVDRegistration(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, target: torch.Tensor, source: torch.Tensor, weights=None):
        isnan = torch.isnan(source) | torch.isnan(target)

        mean_s = weightedNanMean(
            source, weights=weights, axis=-2, keepdims=True)
        mean_t = weightedNanMean(target, weights, axis=-2, keepdims=True)
        xyz_tc = target - mean_t
        xyz_sc = source - mean_s

        if weights is not None:
            xyz_tc = xyz_tc * weights
        else:
            xyz_sc[isnan] = 0
            xyz_tc[isnan] = 0

        cov = xyz_tc.transpose(-1, -2) @ xyz_sc  # 3xn @ nx3
        u, s, vh = torch.linalg.svd(cov)
        rot = u @ vh

        t = mean_t.transpose(-1, -2) - (rot @ mean_s.transpose(-1, -2))
        shape = list(target.shape)
        shape[-2:] = [4, 4]
        transformation = torch.zeros(
            shape, dtype=target.dtype, device=target.device)
        transformation[..., :3, :3] = rot
        transformation[..., :3, -1:] = t
        transformation[..., -1, -1] = 1

        return transformation


class CorrespondenceWeighter(nn.Module):
    def __init__(self, weighting='max'):
        super().__init__()
        self.weighting = weighting
        if weighting == 'topk':
            self.top_k = 20  # FIXME: make parameter
            self.model = nn.Sequential(
                nn.Linear(20, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

        self.weight_fkt = eval('self.'+weighting)  # function name

    def forward(self, weights):
        return (self.weight_fkt(weights))

    def max(self, weights):
        max_v, _ = torch.max(weights, dim=-1, keepdim=True)
        return max_v

    def information_gain(self, p: torch.Tensor):
        dim = torch.sum(p > 0, dim=-1, keepdim=True)
        # KL Divergence between the weights and the uniform distribution
        t = p*(p*dim+1e-8).log()/torch.log(dim)
        return torch.clamp(torch.sum(t, dim=-1, keepdim=True), min=0)

    def information_gain_w(self, p: torch.Tensor):
        w = self.information_gain(p)
        return w/(w.sum(-2, keepdim=True)+1e-8)

    def topk(self, weights):
        top_w = torch.topk(weights, k=self.top_k, dim=-1)[0]
        mask = torch.max(top_w, dim=-1, keepdim=True)[0] > 0
        return self.model(top_w) * mask

    def ones(self, weights):
        return torch.max(weights, dim=-1, keepdim=True)[0] > 0

##############################################
############## KPConv ########################
##############################################


def vector_gather(vectors: torch.Tensor, indices: torch.Tensor):
    """
    Gathers (batched) vectors according to indices.
    Arguments:
        vectors: Tensor[B, N1, D]
        indices: Tensor[B, N2, K]
    Returns:
        Tensor[B,N2, K, D]
    """
    # src
    vectors = vectors.unsqueeze(-2)
    shape = list(vectors.shape)
    shape[-2] = indices.shape[-1]
    vectors = vectors.expand(shape)

    # Do the magic
    shape = list(indices.shape)+[vectors.shape[-1]]
    indices = indices.unsqueeze(-1).expand(shape)
    out = torch.gather(vectors, dim=-3, index=indices)
    return out


def gather(x, idx, method=2):
    """
    implementation of a custom gather operation for faster backwards.
    :param x: input with shape [N, D_1, ... D_d]
    :param idx: indexing with shape [n_1, ..., n_m]
    :param method: Choice of the method
    :return: x[idx] with shape [n_1, ..., n_m, D_1, ... D_d]
    """

    if method == 0:
        return x[idx]
    elif method == 1:
        x = x.unsqueeze(1)
        x = x.expand((-1, idx.shape[-1], -1))
        idx = idx.unsqueeze(2)
        idx = idx.expand((-1, -1, x.shape[-1]))
        return x.gather(0, idx)
    elif method == 2:
        for i, ni in enumerate(idx.size()[1:]):
            x = x.unsqueeze(i+1)
            new_s = list(x.size())
            new_s[i+1] = ni
            x = x.expand(new_s)
        n = len(idx.size())
        for i, di in enumerate(x.size()[n:]):
            idx = idx.unsqueeze(i+n)
            new_s = list(idx.size())
            new_s[i+n] = di
            idx = idx.expand(new_s)
        return x.gather(0, idx)
    else:
        raise ValueError('Unkown method')


def knn(q_pts, s_pts, k):
    dist = ((q_pts.unsqueeze(-2) - s_pts.unsqueeze(-3))**2).sum(-1)
    _, neighb_inds = torch.topk(dist, k, dim=-1, largest=False)
    return neighb_inds


class KPConv(nn.Module):

    def __init__(self, in_channels, out_channels, radius, kernel_size=3, KP_extent=None, p_dim=3, radial=False):
        """
        Initialize parameters for KPConvDeformable.
        :param in_channels: dimension of input features.
        :param out_channels: dimension of output features.
        :param radius: radius used for kernel point init.
        :param kernel_size: Number of kernel points.
        :param KP_extent: influence radius of each kernel point. (float), default: None
        :param p_dim: dimension of the point space. Default: 3
        :param radial: bool if direction independend convolution 
        """
        super(KPConv, self).__init__()

        # Save parameters
        self.radial = radial
        self.p_dim = 1 if radial else p_dim  # 1D for radial convolution

        self.K = kernel_size ** self.p_dim
        self.num_kernels = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.KP_extent = radius / (kernel_size-1) * \
            self.p_dim**0.5 if KP_extent is None else KP_extent

        # Initialize weights
        self.weights = Parameter(torch.zeros((self.K, in_channels, out_channels), dtype=torch.float32),
                                 requires_grad=True)

        # Reset parameters
        self.reset_parameters()

        # Initialize kernel points
        self.kernel_points = self.init_KP()

        return

    def reset_parameters(self):
        kaiming_uniform_(self.weights, a=math.sqrt(5))
        return

    def init_KP(self):
        """
        Initialize the kernel point positions in a grid
        :return: the tensor of kernel points
        """

        K_points_numpy = self.getKernelPoints(self.radius,
                                              self.num_kernels, dim=self.p_dim)

        return Parameter(torch.tensor(K_points_numpy, dtype=torch.float32),
                         requires_grad=False)

    def getKernelPoints(self, radius, num_points=3, dim=3):
        """[summary]

        Args:
            radius (float): radius
            num_points (int, optional): Number of kernel points per dimension. Defaults to 3.

        Returns:
            [type]: returns num_points^3 kernel points 
        """
        xyz = np.linspace(-1, 1, num_points)
        if dim == 1:
            return xyz[:, None]*radius

        points = np.meshgrid(*(dim*[xyz]))
        points = [p.flatten() for p in points]
        points = np.vstack(points).T
        points /= dim**(0.5)  # Normalizes to stay in unit sphere
        return points*radius

    def forward(self, q_pts, s_pts, neighb_inds, x):
        # Add a fake point/feature in the last row for shadow neighbors
        s_pts = torch.cat(
            (s_pts, torch.zeros_like(s_pts[..., :1, :]) + 1e6), -2)
        x = torch.cat((x, torch.zeros_like(x[..., :1, :])), -2)

        # Get neighbor points and features [n_points, n_neighbors, dim/ in_fdim]
        if len(neighb_inds.shape) < 3:
            neighbors = s_pts[neighb_inds, :]
            neighb_x = gather(x, neighb_inds)
        else:
            neighbors = vector_gather(s_pts, neighb_inds)
            neighb_x = vector_gather(x, neighb_inds)

        # Center every neighborhood [n_points, n_neighbors, dim]
        neighbors = neighbors - q_pts.unsqueeze(-2)

        # Get all difference matrices [n_points, n_neighbors, n_kpoints, dim]
        if self.radial:
            neighbors = torch.sqrt(torch.sum(neighbors**2, -1, keepdim=True))
        neighbors.unsqueeze_(-2)

        differences = neighbors - self.kernel_points
        # Get the square distances [n_points, n_neighbors, n_kpoints]
        sq_distances = torch.sum(differences ** 2, dim=-1)
        # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
        all_weights = torch.clamp(
            1 - torch.sqrt(sq_distances) / self.KP_extent, min=0.0)
        # [n_points, n_neighbors, n_kpoints]
        all_weights = torch.transpose(all_weights, -2, -1)

        # Apply distance weights [n_points, n_kpoints, in_fdim]
        weighted_features = all_weights @ neighb_x

        # Apply network weights [n_kpoints, n_points, out_fdim]
        shape = list(range(len(weighted_features.shape)))
        shape[-3:] = (-2, -3, -1)
        weighted_features = weighted_features.permute(shape)

        kernel_outputs = weighted_features @ self.weights

        # Convolution sum [n_points, out_fdim]
        return kernel_outputs.sum(dim=-3)

    def __repr__(self):
        return 'KPConv(radius: {:.2f}, in_feat: {:d}, out_feat: {:d})'.format(self.radius,
                                                                              self.in_channels,
                                                                              self.out_channels)


class ResnetKPConv(nn.Module):
    def __init__(self, in_channels, out_channels, radius, kernel_size=3, KP_extent=None, p_dim=3, radial=False, f_dscale=2):
        super().__init__()

        self.ln1 = nn.LayerNorm(in_channels)
        self.relu = nn.LeakyReLU()

        self.kpconv = KPConv(in_channels=in_channels,
                             out_channels=out_channels//f_dscale,
                             radius=radius,
                             kernel_size=kernel_size,
                             KP_extent=KP_extent,
                             p_dim=p_dim,
                             radial=radial)

        self.ln2 = nn.LayerNorm(out_channels//f_dscale)
        self.lin = nn.Linear(out_channels//f_dscale, out_channels)

        self.in_projection = nn.Identity() if in_channels == out_channels else nn.Linear(
            in_channels, out_channels)

    def forward(self, q_pts, s_pts, neighb_inds, x):
        xr = self.relu(self.ln1(x))
        xr = self.kpconv(q_pts, s_pts, neighb_inds, x)
        xr = self.relu(self.ln2(xr))
        xr = self.lin(xr)
        return self.in_projection(x) + xr
