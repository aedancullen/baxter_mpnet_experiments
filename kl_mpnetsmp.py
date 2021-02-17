import argparse
import torch
import torch.nn as nn
import numpy as np
import yaml
import os
import pickle
from torch.autograd import Variable
import math
from tools.import_tool import fileImport
import time
import sys

from neuralplanner_functions import *
from architectures import *

from tools.obs_data_loader import load_normalized_dataset


start_st_ind = [2,6,7,8,9,10,11,12]
dofs = len(start_st_ind)


def main(args):
    importer = fileImport()
    nsmp = args.nsmp
    pcd_dir = args.pcd_dir
    yaml_dir = args.yaml_dir
    csv_dir = args.csv_dir
    
    
    encoder = Encoder(args.enc_input_size, args.enc_output_size)
    mlp = MLP(args.mlp_input_size, args.mlp_output_size)
    
    device = torch.device('cpu')
    model_path = args.model_path
    mlp.load_state_dict(torch.load(model_path+args.mlp_model_name, map_location=device))
    encoder.load_state_dict(torch.load(model_path+args.enc_model_name, map_location=device))

    if torch.cuda.is_available():
        encoder.cuda()
        mlp.cuda()
        
    
    env_names = []
        
    i = 0
    for filename in os.listdir(yaml_dir):
        if filename.startswith('request'):
            fullname = os.path.join(yaml_dir, filename)
            with open(fullname, 'r') as handle:
                request_data = yaml.load(handle)
            env = 'sample' + str(i)
            
            env_names.append(env)
            arr = np.array(request_data['start_state']['joint_state']['position'])
            start_state = arr[start_st_ind].tolist()
            
            goal_state = []
            for k in range(dofs):
                goal_state.append(request_data['goal_constraints'][0]['joint_constraints'][k]['position'])
    
            i += 1

    clouds = load_normalized_dataset(env_names, pcd_dir, importer)
    
    #obs=torch.from_numpy(obs)

    #en_inp=to_var(obs)
    #h=encoder(en_inp)
    return
    start=np.zeros(dof,dtype=np.float32)
    goal=np.zeros(dof,dtype=np.float32)
    
    start=torch.from_numpy(start)
    goal=torch.from_numpy(goal)
    
    for n in range(smp_total_samples):
        if i < smp_max_neural:
            inp=torch.cat((start,goal,h.data.cpu()))
            inp=to_var(inp)
            inp=mlp(inp)
            inp=inp.data.cpu()
            samples.append(inp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/sample/')
    parser.add_argument('--mlp_model_name', type=str, default='mlp_PReLU_ae_dd140.pkl')
    parser.add_argument('--enc_model_name', type=str, default='cae_encoder_140.pkl')

    parser.add_argument('--enc_input_size', type=int, default=16053)
    parser.add_argument('--enc_output_size', type=int, default=60)
    parser.add_argument('--mlp_input_size', type=int, default=76)
    parser.add_argument('--mlp_output_size', type=int, default=8)
    
    # Number of samples to output
    parser.add_argument('--nsmp', type=int, default=100)
    
    parser.add_argument('--pcd_dir', type=str, default='.')
    parser.add_argument('--yaml_dir', type=str, default='.')
    parser.add_argument('--csv_dir', type=str, default='.')
    
    args = parser.parse_args()
    main(args)
