import argparse
import cPickle as pickle
import os
import numpy as np
import yaml

def main(args):
    pkl_dir = args.pkl_dir
    yaml_dir = args.yaml_dir
    
    trainPaths = {}
    trainEnvironments = {'poses': {}}

    i = 1
    for filename in sorted(os.listdir(yaml_dir)):
        if filename.startswith('path'):
            fullname = os.path.join(yaml_dir, filename)
            with open(fullname, 'r') as handle:
                yaml_data = yaml.load(handle)
            env = 'sample' + str(i)
            
            points = yaml_data['joint_trajectory']['points']
            trainEnvironments['poses'][env] = None
            trainPaths[env] = [np.zeros((len(points), len(points[0]['positions'])))]
            
            for pos_idx, pos_data in enumerate(points):
                trainPaths[env][0][pos_idx, :] = pos_data['positions']
                
            i += 1
            print('Loaded ' + filename)
        
    with open(pkl_dir + '/trainPaths.pkl', 'wb') as handle:
        pickle.dump(trainPaths, handle)
    with open(pkl_dir + '/trainEnvironments.pkl', 'wb') as handle:
        pickle.dump(trainEnvironments, handle)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_dir', type=str, default='.')
    parser.add_argument('--yaml_dir', type=str, default='.')
    
    args = parser.parse_args()
    main(args)
