import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from tools.path_data_loader import load_dataset_end2end
from torch.autograd import Variable
import math
from tools.import_tool import fileImport
import time
import sys
from tensorboardX import SummaryWriter
###
from torch_architectures import *


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def get_input(i, data, targets, pc_inds, obstacles, bs):
    """
    Input:  i (int) - starting index for the batch
            data/targets/pc_inds (numpy array) - data vectors to obtain batch from
            obstacles (numpy array) - point cloud array
            bs (int) - batch size
    """
    if i+bs < len(data):
        bi = data[i:i+bs]
        bt = targets[i:i+bs]
        bpc = pc_inds[i:i+bs]
        bobs = obstacles[bpc]
    else:
        bi = data[i:]
        bt = targets[i:]
        bpc = pc_inds[i:]
        bobs = obstacles[bpc]

    return torch.from_numpy(bi), torch.from_numpy(bt), torch.from_numpy(bobs).type(torch.FloatTensor)


def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (args.device)

    importer = fileImport()
    env_data_path = args.env_data_path
    path_data_path = args.path_data_path
    pcd_data_path = args.pointcloud_data_path

    envs = importer.environments_import(env_data_path + args.envs_file)

    # select how many envs to train on
    envs = envs[:args.N]
    print("Loading obstacle data...\n")
    dataset_train, targets_train, pc_inds_train, obstacles = load_dataset_end2end(
        envs, path_data_path, pcd_data_path, args.path_data_file, importer, NP=args.NP)

    print("Loaded dataset, targets, and pontcloud obstacle vectors: ")
    print(str(len(dataset_train)) + " " +
        str(len(targets_train)) + " " + str(len(pc_inds_train)))
    print("\n")

    if not os.path.exists(args.trained_model_path):
        os.makedirs(args.trained_model_path)

    # Build the models
    mlp = MLP(args.mlp_input_size, args.mlp_output_size)
    if args.AE_type == 'linear':
        encoder = Encoder(args.enc_input_size, args.enc_output_size)
    elif args.AE_type == 'voxel':
        encoder = VoxelEncoder(args.enc_input_size, args.enc_output_size)
        # convert obstacles to voxel
        obstacles = obstacles.astype(float).reshape(len(obstacles),-1,3)
        obstacles = importer.pointcloud_to_voxel(obstacles,
                voxel_size=(args.enc_vox_size, args.enc_vox_size, args.enc_vox_size),
                padding_size=(args.enc_input_size, args.enc_input_size, args.enc_input_size)).reshape(len(obstacles),1,args.enc_input_size,args.enc_input_size,args.enc_input_size)
    elif args.AE_type == 'voxel2':
        encoder = VoxelEncoder2(args.enc_input_size, args.enc_output_size)
        # convert obstacles to voxel
        obstacles = obstacles.astype(float).reshape(len(obstacles),-1,3)
        obstacles = importer.pointcloud_to_voxel(obstacles,
                voxel_size=(args.enc_vox_size, args.enc_vox_size, args.enc_vox_size),
                padding_size=(args.enc_input_size, args.enc_input_size, args.enc_input_size)).reshape(len(obstacles),1,args.enc_input_size,args.enc_input_size,args.enc_input_size)
    elif args.AE_type == 'voxel3':
        encoder = VoxelEncoder3(args.enc_input_size, args.enc_output_size)
        # convert obstacles to voxel
        obstacles = obstacles.astype(float).reshape(len(obstacles),-1,3)
        obstacles = importer.pointcloud_to_voxel(obstacles,
                voxel_size=(args.enc_vox_size, args.enc_vox_size, args.enc_vox_size),
                padding_size=(args.enc_input_size, args.enc_input_size, args.enc_input_size)).reshape(len(obstacles),1,args.enc_input_size,args.enc_input_size,args.enc_input_size)
    elif args.AE_type == 'voxel4':
        encoder = VoxelEncoder4(args.enc_input_size, args.enc_output_size)
        # convert obstacles to voxel
        obstacles = obstacles.astype(float).reshape(len(obstacles),-1,3)
        obstacles = importer.pointcloud_to_voxel(obstacles,
                voxel_size=(args.enc_vox_size, args.enc_vox_size, args.enc_vox_size),
                padding_size=(args.enc_input_size, args.enc_input_size, args.enc_input_size)).reshape(len(obstacles),1,args.enc_input_size,args.enc_input_size,args.enc_input_size)
    elif args.AE_type == 'voxel5':
        encoder = VoxelEncoder5(args.enc_input_size, args.enc_output_size)
        # convert obstacles to voxel
        obstacles = obstacles.astype(float).reshape(len(obstacles),-1,3)
        obstacles = importer.pointcloud_to_voxel(obstacles,
                voxel_size=(args.enc_vox_size, args.enc_vox_size, args.enc_vox_size),
                padding_size=(args.enc_input_size, args.enc_input_size, args.enc_input_size)).reshape(len(obstacles),1,args.enc_input_size,args.enc_input_size,args.enc_input_size)
    elif args.AE_type == 'mvvoxel3':
        encoder = MultiViewVoxelEncoder3(args.enc_input_size, args.enc_output_size)
        # convert obstacles to voxel
        obstacles = obstacles.astype(float).reshape(len(obstacles),-1,3)
        obstacles = importer.pointcloud_to_voxel(obstacles,
                voxel_size=(args.enc_vox_size, args.enc_vox_size, args.enc_vox_size),
                padding_size=(args.enc_input_size, args.enc_input_size, args.enc_input_size)).reshape(len(obstacles),1,args.enc_input_size,args.enc_input_size,args.enc_input_size)
    elif args.AE_type == 'mvvoxel4':
        encoder = MultiViewVoxelEncoder4(args.enc_input_size, args.enc_output_size)
        # convert obstacles to voxel
        obstacles = obstacles.astype(float).reshape(len(obstacles),-1,3)
        obstacles = importer.pointcloud_to_voxel(obstacles,
                voxel_size=(args.enc_vox_size, args.enc_vox_size, args.enc_vox_size),
                padding_size=(args.enc_input_size, args.enc_input_size, args.enc_input_size)).reshape(len(obstacles),1,args.enc_input_size,args.enc_input_size,args.enc_input_size)
    elif args.AE_type == 'mvvoxel5':
        encoder = MultiViewVoxelEncoder5(args.enc_input_size, args.enc_output_size)
        # convert obstacles to voxel
        obstacles = obstacles.astype(float).reshape(len(obstacles),-1,3)
        obstacles = importer.pointcloud_to_voxel(obstacles,
                voxel_size=(args.enc_vox_size, args.enc_vox_size, args.enc_vox_size),
                padding_size=(args.enc_input_size, args.enc_input_size, args.enc_input_size)).reshape(len(obstacles),1,args.enc_input_size,args.enc_input_size,args.enc_input_size)
    elif args.AE_type == 'mvvoxel6':
        encoder = MultiViewVoxelEncoder6(args.enc_input_size, args.enc_output_size)
        # convert obstacles to voxel
        obstacles = obstacles.astype(float).reshape(len(obstacles),-1,3)
        obstacles = importer.pointcloud_to_voxel(obstacles,
                voxel_size=(args.enc_vox_size, args.enc_vox_size, args.enc_vox_size),
                padding_size=(args.enc_input_size, args.enc_input_size, args.enc_input_size)).reshape(len(obstacles),1,args.enc_input_size,args.enc_input_size,args.enc_input_size)
    elif args.AE_type == 'svvoxel1':
        encoder = SingleViewVoxelEncoder1(args.enc_input_size, args.enc_output_size)
        # convert obstacles to voxel
        obstacles = obstacles.astype(float).reshape(len(obstacles),-1,3)
        obstacles = importer.pointcloud_to_voxel(obstacles,
                voxel_size=(args.enc_vox_size, args.enc_vox_size, args.enc_vox_size),
                padding_size=(args.enc_input_size, args.enc_input_size, args.enc_input_size)).reshape(len(obstacles),1,args.enc_input_size,args.enc_input_size,args.enc_input_size)
    elif args.AE_type == 'svvoxel2':
        encoder = SingleViewVoxelEncoder2(args.enc_input_size, args.enc_output_size)
        # convert obstacles to voxel
        obstacles = obstacles.astype(float).reshape(len(obstacles),-1,3)
        obstacles = importer.pointcloud_to_voxel(obstacles,
                voxel_size=(args.enc_vox_size, args.enc_vox_size, args.enc_vox_size),
                padding_size=(args.enc_input_size, args.enc_input_size, args.enc_input_size)).reshape(len(obstacles),1,args.enc_input_size,args.enc_input_size,args.enc_input_size)
    elif args.AE_type == 'svvoxel3':
        encoder = SingleViewVoxelEncoder3(args.enc_input_size, args.enc_output_size)
        # convert obstacles to voxel
        obstacles = obstacles.astype(float).reshape(len(obstacles),-1,3)
        obstacles = importer.pointcloud_to_voxel(obstacles,
                voxel_size=(args.enc_vox_size, args.enc_vox_size, args.enc_vox_size),
                padding_size=(args.enc_input_size, args.enc_input_size, args.enc_input_size)).reshape(len(obstacles),1,args.enc_input_size,args.enc_input_size,args.enc_input_size)
    elif args.AE_type == 'mvvoxel7':
        encoder = MultiViewVoxelEncoder7(args.enc_input_size, args.enc_output_size)
        # convert obstacles to voxel
        obstacles = obstacles.astype(float).reshape(len(obstacles),-1,3)
        obstacles = importer.pointcloud_to_voxel(obstacles,
                voxel_size=(args.enc_vox_size, args.enc_vox_size, args.enc_vox_size),
                padding_size=(args.enc_input_size, args.enc_input_size, args.enc_input_size)).reshape(len(obstacles),1,args.enc_input_size,args.enc_input_size,args.enc_input_size)


    if torch.cuda.is_available():
        encoder.cuda()
        mlp.cuda()


    # Loss and Optimizer
    criterion = nn.MSELoss()
    params = list(encoder.parameters())+list(mlp.parameters())
    optimizer = torch.optim.Adagrad(params, lr=args.learning_rate)

    total_loss = []
    epoch = 1

    sm = 0  # start saving models after 100 epochs

    print("Starting epochs...\n")
    # epoch=1
    done = False
    writer = SummaryWriter('./runs/'+args.exp_name)
    record_i = 0
    record_loss = 0.
    for epoch in range(args.num_epochs):
        # while (not done)
        start = time.time()
        print("epoch" + str(epoch))
        avg_loss = 0
        for i in range(0, len(dataset_train), args.batch_size):
            time_start = time.time()
            print("number of batch: %d/%d" % (i/args.batch_size, len(dataset_train)/args.batch_size))
            # Forward, Backward and Optimize
            # zero gradients
            encoder.zero_grad()
            mlp.zero_grad()

            # convert to pytorch tensors and Varialbes
            bi, bt, bobs = get_input(
                i, dataset_train, targets_train, pc_inds_train, obstacles, args.batch_size)
            bi = to_var(bi)
            bt = to_var(bt)
            bobs = to_var(bobs)

            # forward pass through encoder
            h = encoder(bobs)

            # concatenate encoder output with dataset input
            inp = torch.cat((bi, h), dim=1)

            # forward pass through mlp
            bo = mlp(inp)

            # compute overall loss and backprop all the way
            loss = criterion(bo, bt)
            avg_loss = avg_loss+loss.data
            loss.backward()
            print('loss:')
            print(loss)
            # every 100 batches record loss
            record_loss += loss.data
            record_i += 1
            if record_i % 100 == 0:
                writer.add_scalar('loss', record_loss / 100, record_i)
                record_loss = 0.
            #print('encoder...')
            #for name, param in encoder.named_parameters():
            #    if param.requires_grad:
            #        print name, param.data, param.grad
            #print('mlp...')
            #for name, param in mlp.named_parameters():
            #    if param.requires_grad:
            #        print name, param.data, param.grad
            optimizer.step()
            time_stop = time.time()
            print('training time: %fs\n' % (time_stop - time_start))
            print('estimated remaining time: %fs\n' %
                (((len(dataset_train)-i) / float(args.batch_size) - 1) * (time_stop - time_start) * args.num_epochs))
        print("--average loss:")
        print(avg_loss/(len(dataset_train)/args.batch_size))
        total_loss.append(avg_loss/(len(dataset_train)/args.batch_size))
        # Save the models
        if epoch == sm:
            print("\nSaving model\n")
            print("time: " + str(time.time() - start))
            torch.save(encoder.state_dict(), os.path.join(
                args.trained_model_path, 'cae_encoder_'+str(epoch)+'.pkl'))
            torch.save(total_loss, 'total_loss_'+str(epoch)+'.dat')

            model_path = 'mlp_PReLU_ae_dd'+str(epoch)+'.pkl'
            torch.save(mlp.state_dict(), os.path.join(
                args.trained_model_path, model_path))
            if (epoch != 1):
                sm = sm+20  # save model after every 50 epochs from 100 epoch ownwards

    torch.save(total_loss, 'total_loss.dat')
    model_path = 'mlp_PReLU_ae_dd_final.pkl'
    torch.save(mlp.state_dict(), os.path.join(args.trained_model_path, model_path))

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--env_data_path', type=str, default='./env/environment_data/')
    parser.add_argument('--path_data_path', type=str, default='./data/train/paths/')
    parser.add_argument('--pointcloud_data_path', type=str, default='./data/train/pcd/')
    parser.add_argument('--trained_model_path', type=str, default='./models/sample_train/', help='path for saving trained models')
    parser.add_argument('--exp_name', type=str, default='aa', help='for comparing different experiments')


    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=200)

    parser.add_argument('--enc_vox_size', type=int, default=28)
    parser.add_argument('--enc_input_size', type=int, default=16053)
    parser.add_argument('--enc_output_size', type=int, default=60)
    parser.add_argument('--mlp_input_size', type=int, default=74)
    parser.add_argument('--mlp_output_size', type=int, default=7)

    parser.add_argument('--envs_file', type=str, default='trainEnvironments.pkl')
    parser.add_argument('--path_data_file', type=str, default='trainPaths.pkl')
    parser.add_argument('--AE_type', type=str, default='linear')
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--NP', type=int, default=1000)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    main(args)
