import torch
import torch.nn as nn
import torch.nn.functional as F
from oil.model_trainers import Trainer
from lie_conv.utils import export, Named
import numpy as np

@export
class AutoregressiveMoleculeTrainer(Trainer):
    def __init__(self, *args,**kwargs):
        super().__init__(*args,**kwargs)
            
    def loss(self, mb):
        return self.model.NLL(mb)

    def metrics(self, loader):
        nll = lambda mb: self.model.NLL(mb).cpu().data.numpy()
        return {'NLL': self.evalAverageMetrics(loader,nll)}
    
    # def logStuff(self,step,minibatch=None):
    #     super().logStuff(step,minibatch)                            

