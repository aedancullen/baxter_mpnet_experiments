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

joint_limit_upper = np.array([0.38615, 1.6056,  1.518,  3.1416,  2.251,  3.1416,  2.16,  3.1416])
joint_limit_lower = np.array([0,      -1.6056, -1.221, -3.1416, -2.251, -3.1416, -2.16, -3.1416])
joint_range = joint_limit_upper - joint_limit_lower

def normalize_joints(point):
    return np.divide(point - joint_limit_lower, joint_range, dtype=np.float32)

def rescale_joints(point):
    return np.multiply(point, joint_range, dtype=np.float32) + joint_limit_lower

def main(args):
    importer = fileImport()
    nsmp = args.nsmp
    goal_distance = args.goal_distance
    pcd_dir = args.pcd_dir
    yaml_dir = args.yaml_dir
    csv_dir = args.csv_dir
    
    
    encoder = Encoder(args.enc_input_size, args.enc_output_size)
    mlp = MLP(args.mlp_input_size, args.mlp_output_size)
    
    #device = torch.device('cpu')
    model_path = args.model_path
    mlp.load_state_dict(torch.load(model_path+args.mlp_model_name, ))#map_location=device))
    encoder.load_state_dict(torch.load(model_path+args.enc_model_name, ))#map_location=device))

    if torch.cuda.is_available():
        encoder.cuda()
        mlp.cuda()
        
    
    env_names = []
    start_states = []
    goal_states = []
        
    i = 1
    for filename in sorted(os.listdir(yaml_dir)):
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
    
            start_states.append(goal_state)
            goal_states.append(start_state)
            
            i += 1
            print('Loaded ' + filename)
            if i == 101:
                break

    clouds = load_normalized_dataset(env_names, pcd_dir, importer)
    
    
    for i in range(len(env_names)):
        start_state = start_states[i]
        goal_state = goal_states[i]
        cloud = clouds[i]
        
        samples = []
        
        start_array = normalize_joints(np.array(start_state, dtype=np.float32))
        goal_array = normalize_joints(np.array(goal_state, dtype=np.float32))

        current = torch.from_numpy(start_array)
        goal = torch.from_numpy(goal_array)
        
        cloud = torch.from_numpy(cloud)
        en_inp = to_var(cloud)
        h = encoder(en_inp)
        
        for n in range(nsmp):
            inp = torch.cat((current, goal, h.data.cpu()))
            inp = to_var(inp)
            current = mlp(inp)
            current = current.data.cpu()
            current_array = np.array(current, dtype=np.float32)
            samples.append(rescale_joints(current_array))
            
            #if np.linalg.norm(rescale_joints(current_array) - rescale_joints(goal_array)) < goal_distance:
            if n % 350 == 0:
                print("reset")
                current = torch.from_numpy(start_array)
            
        csv_filename = 'precomputed' + str(i + 1) + '.csv'
        with open(csv_dir + '/' + csv_filename, 'w') as handle:
            for sample in samples:
                for idx, dim in enumerate(sample):
                    if idx != 0:
                        handle.write(',')
                    handle.write(str(float(dim)))
                    if idx == len(sample)-1:
                        handle.write('\n')
                        
        print('Generated ' + csv_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models-dropout0.001-backwards-5k/')
    parser.add_argument('--mlp_model_name', type=str, default='mlp_PReLU_ae_dd140.pkl')
    parser.add_argument('--enc_model_name', type=str, default='cae_encoder_140.pkl')

    parser.add_argument('--enc_input_size', type=int, default=16053)
    parser.add_argument('--enc_output_size', type=int, default=60)
    parser.add_argument('--mlp_input_size', type=int, default=76)
    parser.add_argument('--mlp_output_size', type=int, default=8)
    
    # Number of samples to output
    parser.add_argument('--nsmp', type=int, default=50000)
    parser.add_argument('--goal_distance', type=float, default=2)
    
    parser.add_argument('--pcd_dir', type=str, default='.')
    parser.add_argument('--yaml_dir', type=str, default='.')
    parser.add_argument('--csv_dir', type=str, default='.')
    
    args = parser.parse_args()
    main(args)
