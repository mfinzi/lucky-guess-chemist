import torch
import torch.nn as nn
from model import LieConvSimple,simple_lift
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
class MoleculeTrainer(Trainer):
    def __init__(self, *args, task='cv', ds_stats=None, **kwargs):
        super().__init__(*args,**kwargs)
        self.hypers['task'] = task
        self.ds_stats = ds_stats
        if hasattr(self.lr_schedulers[0],'setup_metrics'): #setup lr_plateau if exists
            self.lr_schedulers[0].setup_metrics(self.logger,'valid_MAE')
            
    def loss(self, minibatch):
        y = self.model(minibatch)
        target = minibatch[self.hypers['task']]

        if self.ds_stats is not None:
            median, mad = self.ds_stats
            target = (target - median) / mad

        return (y-target).abs().mean()

    def metrics(self, loader):
        task = self.hypers['task']

        #mse = lambda mb: ((self.model(mb)-mb[task])**2).mean().cpu().data.numpy()
        if self.ds_stats is not None:
            median, mad = self.ds_stats
            def mae(mb):
                target = mb[task]
                y = self.model(mb) * mad + median
                return (y-target).abs().mean().cpu().data.numpy()
        else:
            mae = lambda mb: (self.model(mb)-mb[task]).abs().mean().cpu().data.numpy()
        return {'MAE': self.evalAverageMetrics(loader,mae)}
    
    def logStuff(self,step,minibatch=None):
        super().logStuff(step,minibatch)                            

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