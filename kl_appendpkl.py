import argparse
import pickle
import os
import numpy as np
import yaml

def main(args):
    pkl_file = args.pkl_file
    env_name = args.env_name
    yaml_dir = args.yaml_dir
    
    try:
        with open(pkl_file, "rb") as handle:
            pkl = pickle.load(handle)
    except:
        pkl = {}
        
    pkl[env_name] = []
    

    for filename in os.listdir(yaml_dir):
        if filename.startswith("path"):
            fullname = os.path.join(yaml_dir, filename)
            with open(fullname, 'r') as handle:
                yaml_data = yaml.load(handle)
                
            points = yaml_data['joint_trajectory']['points']
            pkl[env_name].append(np.zeros((len(points), len(points[0]['positions']))))
            
            for pos_idx, pos_data in enumerate(points):
                pkl[env_name][-1][pos_idx, :] = pos_data['positions']


    print(pkl)
        
    with open(pkl_file, "wb") as handle:
        pickle.dump(pkl, handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_file', type=str)
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--yaml_dir', type=str)
    
    args = parser.parse_args()
    main(args)
