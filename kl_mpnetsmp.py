import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from torch.autograd import Variable
import math
import time
import sys

from neuralplanner_functions import *
from architectures import *


def main(args):
    pcd_data_path = args.pointcloud_data_path
    smp_total_samples = args.smp_total_samples
    smp_max_neural = args.smp_max_neural
    
    encoder = Encoder_End2End(args.enc_input_size, args.enc_output_size)
    mlp = MLP(args.mlp_input_size, args.mlp_output_size)
    
    device = torch.device('cpu')
    model_path = args.model_path
    mlp.load_state_dict(torch.load(model_path+args.mlp_model_name, map_location=device))
    encoder.load_state_dict(torch.load(model_path+args.enc_model_name, map_location=device))

    if torch.cuda.is_available():
        encoder.cuda()
        mlp.cuda()

    #obs=torch.from_numpy(obs)

    #en_inp=to_var(obs)
    #h=encoder(en_inp)

    start=np.zeros(dof,dtype=np.float32)
    goal=np.zeros(dof,dtype=np.float32)
    
    start=torch.from_numpy(start)
    goal=torch.from_numpy(goal)
    
    # Qureshi 1907.06013 Algorithm 6 (MPNetSMP)
    for n in range(smp_total_samples):
        if i < smp_max_neural:
            inp=torch.cat((start,goal,h.data.cpu()))
            inp=to_var(inp)
            start=mlp(inp)
            start=start.data.cpu()
            samples.append(start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/sample/')
    parser.add_argument('--mlp_model_name', type=str, default='mlp_PReLU_ae_dd140.pkl')
    parser.add_argument('--enc_model_name', type=str, default='cae_encoder_140.pkl')

    parser.add_argument('--pointcloud_data_path', type=str, default='./data/test/pcd/')

    parser.add_argument('--enc_input_size', type=int, default=16053)
    parser.add_argument('--enc_output_size', type=int, default=60)
    parser.add_argument('--mlp_input_size', type=int, default=74)
    parser.add_argument('--mlp_output_size', type=int, default=7)
    
    # smp_total_samples is "n" in Algorithm 6, smp_max_neural is "Nsmp"
    parser.add_argument('--smp_total_samples', type=int, default=7)
    parser.add_argument('--smp_max_neural', type=int, default=7)
    
    args = parser.parse_args()
    main(args)
