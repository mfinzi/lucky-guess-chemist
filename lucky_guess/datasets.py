import os
import torch
from corm_data.utils import initialize_datasets
default_qm9_dir = '~/datasets/molecular/qm9/'
def QM9datasets(root_dir=default_qm9_dir):
    root_dir = os.path.expanduser(root_dir)
    filename= f"{root_dir}data_boxed.pz"
    if os.path.exists(filename):
        return torch.load(filename)
    else:
        datasets, num_species, charge_scale = initialize_datasets((-1,-1,-1),
         "data", 'qm9', subtract_thermo=True,force_download=True)
        qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114, 'lumo': 27.2114}
        u_bounds = datasets['train'].data['positions'].reshape(-1,3).max(dim=0)[0]
        l_bounds = datasets['train'].data['positions'].reshape(-1,3).min(dim=0)[0]
        for dataset in datasets.values():
            dataset.convert_units(qm9_to_eV)
            dataset.num_species = 5
            dataset.charge_scale = 9
            # updated the xyz atom coordinates so they lie within the box [0-1]^3
            dataset.data['positions'] = .99*(dataset.data['positions']-l_bounds)/(u_bounds-l_bounds)
    
        os.makedirs(root_dir, exist_ok=True)
        torch.save((datasets, num_species, charge_scale),filename)
        return (datasets, num_species, charge_scale)