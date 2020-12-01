import torch
import torch.nn as nn
import torch.nn.functional as F
from lucky_guess.lieconv_layers import LieConvSimple,simple_lift,LieConvAutoregressive
from oil.model_trainers import Trainer
from lie_conv.lieConv import PointConv, Pass, Swish, GlobalPool
from lie_conv.lieConv import norm, LieResNet, BottleBlock
from lie_conv.utils import export, Named
from lie_conv.datasets import SO3aug, SE3aug
from lie_conv.lieGroups import SE3
from lie_conv.utils import Expression,export,Named, Pass
from lie_conv.utils import FarthestSubsample, knn_point, index_points
from lie_conv.lieGroups import T,SO2,SO3,SE2,SE3, norm
from lie_conv.masked_batchnorm import MaskBatchNormNd
import numpy as np

@export
class LieResNet(nn.Module,metaclass=Named):
    """ Generic LieConv architecture from Fig 5. Relevant Arguments:
        [Fill] specifies the fraction of the input which is included in local neighborhood. 
                (can be array to specify a different value for each layer)
        [nbhd] number of samples to use for Monte Carlo estimation (p)
        [chin] number of input channels: 1 for MNIST, 3 for RGB images, other for non images
        [ds_frac] total downsampling to perform throughout the layers of the net. In (0,1)
        [num_layers] number of BottleNeck Block layers in the network
        [k] channel width for the network. Can be int (same for all) or array to specify individually.
        [liftsamples] number of samples to use in lifting. 1 for all groups with trivial stabilizer. Otherwise 2+
        [Group] Chosen group to be equivariant to.
        [bn] whether or not to use batch normalization. Recommended in all cases except dynamical systems.
        """
    def __init__(self, chin, num_outputs=1, k=1536, nbhd=np.inf,
                bn=True, num_layers=6, mean=True,liftsamples=1, fill=1/4, group=SE3, **kwargs):
        super().__init__()
        if isinstance(fill,(float,int)):
            fill = [fill]*num_layers
        if isinstance(k,int):
            k = [k]*(num_layers+1)
        conv = lambda ki,ko,fill: LieConvSimple(ki, ko, mc_samples=nbhd, bn=bn, mean=mean,
                                group=group,fill=fill,**kwargs)
        self.net = nn.Sequential(
            Pass(nn.Linear(chin,k[0]),dim=1), #embedding layer
            *[BottleBlock(k[i],k[i+1],conv,bn=bn,fill=fill[i]) for i in range(num_layers)],
            MaskBatchNormNd(k[-1]) if bn else nn.Sequential(),
            Pass(Swish(),dim=1),
            Pass(nn.Linear(k[-1],num_outputs),dim=1),
            GlobalPool(mean=mean),
            )
        self.liftsamples = liftsamples
        self.group = group

    def forward(self, x):
        lifted_x = simple_lift(self.group,x,self.liftsamples)
        return self.net(lifted_x)

@export 
class MolecLieResNet(LieResNet):
    def __init__(self, num_species, charge_scale, aug=False, group=SE3, **kwargs):
        super().__init__(chin=3*num_species,num_outputs=1,group=group,ds_frac=1,**kwargs)
        self.charge_scale = charge_scale
        self.aug =aug
        self.random_rotate = SE3aug()#RandomRotation()
    def featurize(self, mb):
        charges = mb['charges'] / self.charge_scale
        c_vec = torch.stack([torch.ones_like(charges),charges,charges**2],dim=-1) # 
        one_hot_charges = (mb['one_hot'][:,:,:,None]*c_vec[:,:,None,:]).float().reshape(*charges.shape,-1)
        atomic_coords = mb['positions'].float()
        atom_mask = mb['charges']>0
        #print('orig_mask',atom_mask[0].sum())
        return (atomic_coords, one_hot_charges, atom_mask)
    def forward(self,mb):
        with torch.no_grad():
            x = self.featurize(mb)
            x = self.random_rotate(x) if self.aug else x
        return super().forward(x).squeeze(-1)




@export
class SeqMolec(nn.Module,metaclass=Named):
    """ Generic LieConv architecture from Fig 5. Relevant Arguments:
        [Fill] specifies the fraction of the input which is included in local neighborhood. 
                (can be array to specify a different value for each layer)
        [nbhd] number of samples to use for Monte Carlo estimation (p)
        [chin] number of input channels: 1 for MNIST, 3 for RGB images, other for non images
        [ds_frac] total downsampling to perform throughout the layers of the net. In (0,1)
        [num_layers] number of BottleNeck Block layers in the network
        [k] channel width for the network. Can be int (same for all) or array to specify individually.
        [liftsamples] number of samples to use in lifting. 1 for all groups with trivial stabilizer. Otherwise 2+
        [Group] Chosen group to be equivariant to.
        [bn] whether or not to use batch normalization. Recommended in all cases except dynamical systems.
        """
    
    def __init__(self, num_species, k=1536, nbhd=np.inf,
                     mean=True, num_layers=6,liftsamples=1, fill=1., group=T(3),aug=False, **kwargs):
        super().__init__()
        conv = lambda ki,ko,fill: LieConvAutoregressive(ki, ko, mc_samples=nbhd, bn=True, mean=mean,
                                group=group,fill=fill,**kwargs)
        self.body = nn.Sequential(
            Pass(nn.Linear(num_species,k),dim=1), #embedding layer
            *[BottleBlock(k,k,conv,bn=True,fill=fill) for i in range(num_layers)],
            MaskBatchNormNd(k),
            Pass(Swish(),dim=1),
            Pass(nn.Linear(k,num_species+k),dim=1),
            ) # Choose the atom -> then choose location
        self.position_head = nn.Sequential(
            MaskBatchNormNd(k+num_species),
            Pass(Swish(),dim=1),
            Pass(nn.Linear(k+num_species,3),dim=1),
        )
        self.NUM_SPECIES=num_species
        self.aug=aug
        self.liftsamples = liftsamples
        self.group = group
        self.k=k

    def featurize(self, mb):
        atomic_coords = mb['positions'].float()
        
        com = atomic_coords.mean(1,keepdims=True)
        coords_wcomstart = torch.cat([com,atomic_coords],dim=1)

        bs,n,c = mb['one_hot'].shape
        assert 1+c==self.NUM_SPECIES, print(c,self.NUM_SPECIES)
        one_hots_wstart = torch.zeros(bs,n+1,self.NUM_SPECIES,device=com.device)
        one_hots_wstart[:,1:,1:] = mb['one_hot'].float()
        one_hots_wstart[:,0,0]=1 # start token

        atom_mask = (mb['charges']>0).float() # Don't add start token to mask?
        atom_mask_wstart = torch.cat([torch.zeros(bs,1,device=com.device),atom_mask],dim=1)
        return (coords_wcomstart, one_hots_wstart, atom_mask_wstart>0)

    def NLL(self,mb):
        with torch.no_grad():
            x = self.featurize(mb)
        xyz_in,atom_in,mask_in = x
        bs = xyz_in.shape[0]
        xyz_out,feats_out,mask_out = self.forward(x)
        logits,feats = torch.split(feats_out,[self.NUM_SPECIES,self.k],-1)
        logits_wstart  = torch.roll(logits,1,1)
        atom_index = atom_in.reshape(-1,self.NUM_SPECIES).max(dim=1)[1]
        NLLs = F.cross_entropy(logits_wstart.reshape(-1,self.NUM_SPECIES),atom_index,reduction='none')
        NLL_sum = (NLLs*mask_in.reshape(-1)).sum()
        head_input = (xyz_out,torch.cat([atom_in,torch.roll(feats,1,1)],dim=-1),mask_out)
        _,xyz_pred,mask_out = self.position_head(head_input)
        return (((xyz_pred-xyz_in)**2).sum()/2+NLL_sum)/bs
        
    def forward(self,x):
        with torch.no_grad():
            #x = self.random_rotate(x) if self.aug else x
            lifted_x = simple_lift(self.group,x,self.liftsamples)
        return self.body(lifted_x)
