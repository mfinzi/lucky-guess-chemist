import torch

from lie_conv.lieConv import PointConv
from lie_conv.lieGroups import T,Trivial,SE3, LieGroup
import types

def simple_lift(self,inp,nsamples,**kwargs):
    """assumes x has shape (*,n,d), vals has shape (*,n,c), mask has shape (*,n)
        returns (a,v) with shapes [(*,n*nsamples,lie_dim),(*,n*nsamples,c)"""
    x,v,m = inp
    expanded_a,expanded_q = self.lifted_elems(x,nsamples,**kwargs) # (bs,n*ns,d), (bs,n*ns,qd)
    nsamples = expanded_a.shape[-2]//m.shape[-1]
    # expand v and mask like q
    expanded_v = v[...,None,:].repeat((1,)*len(v.shape[:-1])+(nsamples,1)) # (bs,n,c) -> (bs,n,1,c) -> (bs,n,ns,c)
    expanded_v = expanded_v.reshape(*expanded_a.shape[:-1],v.shape[-1]) # (bs,n,ns,c) -> (bs,n*ns,c)
    expanded_mask = m[...,None].repeat((1,)*len(v.shape[:-1])+(nsamples,)) # (bs,n) -> (bs,n,ns)
    expanded_mask = expanded_mask.reshape(*expanded_a.shape[:-1]) # (bs,n,ns) -> (bs,n*ns)
    # concatenate lie algebra element with orbit identifier
    expanded_aq = torch.cat([expanded_a,expanded_q],dim=-1) if expanded_q is not None else expanded_a
    return (expanded_aq,expanded_v,expanded_mask)

LieGroup.lift = types.MethodType(simple_lift,LieGroup)

class LieConvSimple(PointConv):
    def __init__(self,*args,group=T(3),ds_frac=1,fill=1/3,**kwargs):
        kwargs.pop('xyz_dim',None)
        super().__init__(*args,xyz_dim=group.lie_dim+2*group.q_dim,**kwargs)
        self.group = group # Equivariance group for LieConv
        self.register_buffer('r',torch.tensor(2.)) # Internal variable for local_neighborhood radius, set by fill
        self.fill_frac = min(fill,1.) # Average Fraction of the input which enters into local_neighborhood, determines r
        self.coeff = .5  # Internal coefficient used for updating r
        self.fill_frac_ema = fill # Keeps track of average fill frac, used for logging only
        
    def extract_conv_args(self,algebra_orbits1,algebra_orbits2):
        """ inputs: [aq1 (bs,n,d), aq2 (bs,n,d)] outputs: a12q1q2 (bs,n,n,d2)"""
        a1 = algebra_orbits1[...,:self.group.lie_dim]
        q1 = algebra_orbits1[...,self.group.lie_dim:]
        a2 = algebra_orbits2[...,:self.group.lie_dim]
        q2 = algebra_orbits2[...,self.group.lie_dim:]
        a12 = self.group.log(self.group.exp(-a1).unsqueeze(-4)@self.group.exp(a2).unsqueeze(-3)) # check ordering and signs
        return torch.cat([a12,q1,q2],dim=-1)

    def extract_neighborhood(self,inp,query_aq):
        """ inputs: [aq (bs,n,d), inp_vals (bs,n,c), mask (bs,n), query_indices (bs,m)]
            outputs: [neighbor_abq (bs,m,mc_samples,d), neighbor_vals (bs,m,mc_samples,c)]"""

        # Subsample pairs_ab, inp_vals, mask to the query_indices
        aq, inp_vals, mask = inp
        abq_at_query = self.extract_conv_args(aq,query_aq)
        
        # Determine ids (and mask) for points sampled within neighborhood (A4)
        dists = self.group.distance(abq_at_query) #(bs,m,n,d) -> (bs,m,n)
        dists = torch.where(mask[:,None,:].expand(*dists.shape),dists,1e8*torch.ones_like(dists))
        k = min(self.mc_samples,inp_vals.shape[1])
        bs,m,n = dists.shape
        within_ball = (dists < self.r)&mask[:,None,:]&mask[:,:,None] # (bs,m,n)
        B = torch.arange(bs)[:,None,None]
        M = torch.arange(m)[None,:,None]
        noise = torch.zeros(bs,m,n,device=within_ball.device)
        noise.uniform_(0,1)
        valid_within_ball, nbhd_idx =torch.topk(within_ball+noise,k,dim=-1,largest=True,sorted=False)
        valid_within_ball = (valid_within_ball>1)
        
        # Retrieve abq_pairs, values, and mask at the nbhd locations
        B = torch.arange(inp_vals.shape[0],device=inp_vals.device).long()[:,None,None].expand(*nbhd_idx.shape)
        M = torch.arange(abq_at_query.shape[1],device=inp_vals.device).long()[None,:,None].expand(*nbhd_idx.shape)
        nbhd_abq = abq_at_query[B,M,nbhd_idx]     #(bs,m,n,d) -> (bs,m,mc_samples,d)
        nbhd_vals = inp_vals[B,nbhd_idx]   #(bs,n,c) -> (bs,m,mc_samples,c)
        nbhd_mask = mask[B,nbhd_idx]            #(bs,n) -> (bs,m,mc_samples)
        
        if self.training: # update ball radius to match fraction fill_frac inside
            navg = (within_ball.float()).sum(-1).sum()/mask[:,:,None].sum()
            avg_fill = (navg/mask.sum(-1).float().mean()).cpu().item()
            self.r +=  self.coeff*(self.fill_frac - avg_fill)
            self.fill_frac_ema += .1*(avg_fill-self.fill_frac_ema)
        return nbhd_abq, nbhd_vals, (nbhd_mask&valid_within_ball.bool())

    # def log_data(self,logger,step,name):
    #     logger.add_scalars('info', {f'{name}_fill':self.fill_frac_ema}, step=step)
    #     logger.add_scalars('info', {f'{name}_R':self.r}, step=step)

    def point_convolve(self,embedded_group_elems,nbhd_vals,nbhd_mask):
        """ Uses generalized PointConv trick (A1) to compute convolution using pairwise elems (aij) and nbhd vals (vi).
            inputs [embedded_group_elems (bs,m,mc_samples,d), nbhd_vals (bs,m,mc_samples,ci), nbhd_mask (bs,m,mc_samples)]
            outputs [convolved_vals (bs,m,co)]"""
        bs, m, nbhd, ci = nbhd_vals.shape  # (bs,m,mc_samples,d) -> (bs,m,mc_samples,cm*co/ci)
        _, penult_kernel_weights, _ = self.weightnet((None,embedded_group_elems,nbhd_mask))
        penult_kernel_weights_m = torch.where(nbhd_mask.unsqueeze(-1),penult_kernel_weights,torch.zeros_like(penult_kernel_weights))
        nbhd_vals_m = torch.where(nbhd_mask.unsqueeze(-1),nbhd_vals,torch.zeros_like(nbhd_vals))
        #      (bs,m,mc_samples,ci) -> (bs,m,ci,mc_samples) @ (bs, m, mc_samples, cmco/ci) -> (bs,m,ci,cmco/ci) -> (bs,m, cmco) 
        partial_convolved_vals = (nbhd_vals_m.transpose(-1,-2)@penult_kernel_weights_m).view(bs, m, -1)
        convolved_vals = self.linear(partial_convolved_vals) #  (bs,m,cmco) -> (bs,m,co)
        if self.mean: convolved_vals /= nbhd_mask.sum(-1,keepdim=True).clamp(min=1) # Divide by num points
        return convolved_vals

    def forward(self, inp):
        """inputs: [pairs_abq (bs,n,n,d)], [inp_vals (bs,n,ci)]), [query_indices (bs,m)]
           outputs [subsampled_abq (bs,m,m,d)], [convolved_vals (bs,m,co)]"""
        nbhd_abq, nbhd_vals, nbhd_mask = self.extract_neighborhood(inp, query_indices)
        convolved_vals = self.point_convolve(nbhd_abq, nbhd_vals, nbhd_mask)
        convolved_wzeros = torch.where(sub_mask.unsqueeze(-1),convolved_vals,torch.zeros_like(convolved_vals))
        return sub_abq, convolved_wzeros, sub_mask
