'''
usd cmd:
python dataset.py --data_path=./data/material_project --atom_feature=atom_onehot.json --graph_cache=graph_cache.pickle --save_path=elastic_ofm.pickle --property elastic.csv

python dataset.py --data_path=./data/material_project --atom_feature=atom_onehot.json --graph_cache=graph_cache.pickle --save_path=mp_ofm.pickle --property property.csv

python dataset.py --data_path=./data/jarvis --atom_feature=atom_onehot.json --graph_cache=graph_cache.pickle --save_path=jarvis_full_onehot.pickle --property jarvis_full.csv

'''

import argparse
import os
import numpy as np
import csv
import pickle
import re

from tqdm import tqdm
from argparse import Namespace

from chemprop.data import CrystalDatapoint, CrystalDataset
from chemprop.features import AtomCustomJSONInitializer, GaussianDistance, load_radius_dict
from chemprop.data.scaler import StandardScaler

def fatch_sapce_feature(path: str):
    # :param path: path (including the file name) containing the features
    # :return dict(name -> array(unified))
    space_feature = np.loadtxt(path, dtype=str, delimiter=',', skiprows=1)
    space_vector = space_feature[:, 1:].astype(float)
    # print(space_feature)
    # print(np.shape(space_vector))

    scaler = StandardScaler().fit(space_vector)
    scaled_vector = scaler.transform(space_vector).tolist()
    # print(np.shape(scaled_vector))
    # print(scaled_vector)

    space_dict = {}
    for i in range(np.shape(space_feature)[0]):
        space_dict["POSCAR." + space_feature[i][0]] = scaled_vector[i]
    return space_dict
            


def generate_pickle(path: str,
             save_name: str,
             space_file: str,
             property_file: str,
             graph: dict = None,
             ari: object = None,
             gdf: object = None,
             radius_dic: object = None,
             args: Namespace = None):
    # :param path: Path to a CSV file. This CSV file contains all the data (train valid test)
    # :param save_name: the saving file name
    # :param graph: Path to a graph dict.
    # :param args: Arguments.

    # space_feature = fatch_sapce_feature(os.path.join(path, space_file))

    # This is a dict of name -> properties
    crystal_dict = dict()
    with open(os.path.join(path, property_file)) as f:
        reader = list(csv.reader(f))
        for line in reader[1:]:
            crystal_dict[line[0]] = [float(target) for target in line[1:]]

    dataPoint_dict = dict()
    # construct dataset
    # build a dict of name->CrystalDatapoint

    elements = {'H':['1s2'],'Li':['[He] 1s2'],'Be':['[He] 2s2'],'B':['[He] 2s2 2p1'],'N':['[He] 2s2 2p3'],'O':['[He] 2s2 2p4'],
                     'C':['[He] 2s2 2p2'], 'I':['[Kr] 4d10 5s2 5p5'],
                     'F':['[He] 2s2 2p5'],'Na':['[Ne] 3s1'],'Mg':['[Ne] 3s2'],'Al':['[Ne] 3s2 3p1'],'Si':['[Ne] 3s2 3p2'],
                     'P':['[Ne] 3s2 3p3'],'S':['[Ne] 3s2 3p4'],'Cl':['[Ne] 3s2 3p5'],'K':['[Ar] 4s1'],'Ca':['[Ar] 4s2'],'Sc':['[Ar] 3d1 4s2'],
                     'Ti':['[Ar] 3d2 4s2'],'V':['[Ar] 3d3 4s2'],'Cr':['[Ar] 3d5 4s1'],'Mn':['[Ar] 3d5 4s2'],
                     'Fe':['[Ar] 3d6 4s2'],'Co':['[Ar] 3d7 4s2'],'Ni':['[Ar] 3d8 4s2'],'Cu':['[Ar] 3d10 4s1'],'Zn':['[Ar] 3d10 4s2'],
                     'Ga':['[Ar] 3d10 4s2 4p2'],'Ge':['[Ar] 3d10 4s2 4p2'],'As':['[Ar] 3d10 4s2 4p3'],'Se':['[Ar] 3d10 4s2 4p4'],'Br':['[Ar] 3d10 4s2 4p5'],'Rb':['[Kr] 5s1'],
                     'Sr':['[Kr] 5s2'],'Y':['[Kr] 4d1 5s2'],'Zr':['[Kr] 4d2 5s2'],'Nb':['[Kr] 4d4 5s1'],'Mo':['[Kr] 4d5 5s1'],
                     'Ru':['[Kr] 4d7 5s1'],'Rh':['[Kr] 4d8 5s1'],'Pd':['[Kr] 4d10'],'Ag':['[Kr] 4d10 5s1'],'Cd':['[Kr] 4d10 5s2'],
                     'In':['[Kr] 4d10 5s2 5p1'],'Sn':['[Kr] 4d10 5s2 5p2'],'Sb':['[Kr] 4d10 5s2 5p3'],'Te':['[Kr] 4d10 5s2 5p4'],'Cs':['[Xe] 6s1'],'Ba':['[Xe] 6s2'],
                     'La':['[Xe] 5d1 6s2'],'Ce':['[Xe] 4f1 5d1 6s2'],'Hf':['[Xe] 4f14 5d2 6s2'],'Ta':['[Xe] 4f14 5d3 6s2'],
                     'W':['[Xe] 4f14 5d5 6s1'],'Re':['[Xe] 4f14 5d5 6s2'],'Os':['[Xe] 4f14 5d6 6s2'],
                     'Ir':['[Xe] 4f14 5d7 6s2'],'Pt':['[Xe] 4f14 5d10'],'Au':['[Xe] 4f14 5d10 6s1'],'Hg':['[Xe] 4f14 5d10 6s2'],
                     'Tl':['[Xe] 4f14 5d10 6s2 6p2'],'Pb':['[Xe] 4f14 5d10 6s2 6p2'],'Bi':['[Xe] 4f14 5d10 6s2 6p3'],
                     'Tc':['[Kr] 4d5 5s2'],'Fr':['[Rn]7s1'],'Ra':['[Rn]7s2'],'Pr':['[Xe]4f3 6s2'],
                     'Nd':['[Xe] 4f4 6s2'],'Pm':['[Xe] 4f5 6s2'],'Sm':['[Xe] 4f6 6s2'],
                     'Eu':['[Xe] 4f7 6s2'],'Gd':['[Xe] 4f7 5d1 6s2'],'Tb':['[Xe] 4f9 6s2'],
                     'Dy':['[Xe] 4f10 6s2'],'Ho':['[Xe] 4f11 6s2'],'Er':['[Xe] 4f12 6s2'],
                     'Tm':['[Xe] 4f13 6s2'],'Yb':['[Xe] 4f14 6s2'],'Lu':['[Xe] 4f14 5d1 6s2'],
                     'Po':['[Xe] 4f14 5d10 6s2 6p4'],'At':['[Xe] 4f14 5d10 6s2 6p5'],
                     'Ac':['[Rn] 6d1 7s2'],'Th':['[Rn] 6d2 7s2'],'Pa':['[Rn] 5f2 6d1 7s2'],
                     'U':['[Rn] 5f3 6d1 7s2'],'Np':['[Rn] 5f4 6d1 7s2'],'Pu':['[Rn] 5f6 7s2'],
                     'Am':['[Rn] 5f7 7s2'],'Cm':['[Rn] 5f7 6d1 7s2'],'Bk':['[Rn] 5f9 7s2'],
                     'Cf':['[Rn] 5f10 7s2'],'Es':['[Rn] 5f11 7s2'],'Fm':['[Rn] 5f12 7s2'],
                     'Md':['[Rn] 5f13 7s2'],'No':['[Rn] 5f14 7s2'],'Lr':['[Rn] 5f14 6d1 7s2'],
                     'Rf':['[Rn] 5f14 6d2 7s2'],'Db':['[Rn] 5f14 6d3 7s2'],
                     'Sg':['[Rn] 5f14 6d4 7s2'],'Bh':['[Rn] 5f14 6d5 7s2'],
                     'Hs':['[Rn] 5f14 6d6 7s2'],'Mt':['[Rn] 5f14 6d7 7s2'],'Xe': ['[Kr] 4d10 5s2 5p6'], 'He':['1s2'], 'Kr':['[Ar] 3d10 4s2 4p6'], 'Ar': ['[Ne] 3s2 3p6'], 'Ne':['[He] 2s2 2p6']}
    orbitals = {"s1":0,"s2":1,"p1":2,"p2":3,"p3":4,"p4":5,"p5":6,"p6":7,"d1":8,"d2":9,"d3":10,"d4":11,
            "d5":12,"d6":13,"d7":14,"d8":15,"d9":16,"d10":17,"f1":18,"f2":19,"f3":20,"f4":21,
            "f5":22,"f6":23,"f7":24,"f8":25,"f9":26,"f10":27,"f11":28,"f12":29,"f13":30,"f14":31}
    
    hvs = {}
    for key in elements.keys():
        element = key
        hv = np.zeros(shape=(32,1))
        s = elements[key][0]
        sp = (re.split('(\s+)', s))
        if key == "H":
            hv[0] = 1
        if key != "H":
            for j in range(1,len(sp)):
                if sp[j] != ' ':
                    n = sp[j][:1]
                    orb = sp[j][1:]
                    hv[orbitals[orb]] = 1
        hvs[element] = hv


    for name, targets in tqdm(crystal_dict.items(), total=len(crystal_dict)):
        # if i != 66149:
        #     i += 1
        #     continue
        # print(f"name {name}")
        # print(f"target {targets}")
        dataPoint_dict[name] = \
            CrystalDatapoint(
                crystal_name=name,
                crystal_dict=graph[name],
                # space_feature = space_feature[name],
                targets=targets,
                ari=ari,
                gdf=gdf,
                radius_dic=radius_dic,
                hvs=hvs,
                args=args

            )

    with open(os.path.join(path, f'{save_name}'), 'wb') as f:
        pickle.dump(dataPoint_dict, f)


#PICK UP HERE
# add a preprocess process to generate a whole pickle file
# the file contain the whole class of CrystalDataset
# save 'dataset.pickle', dict(name -> CrystalDatapoint)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="specify the data set path", default="./data/material_project", type=str)
    parser.add_argument('--radius', type=int, default=5, help='The crystal neighbor radius')
    parser.add_argument('--max_num_neighbors', type=int, default=8, help='the maximum of crystal neighbors')
    parser.add_argument('--atom_feature', type=str, default='atom_init.json', help='the initial atom embedding')
    parser.add_argument('--save_path', type=str, default='dataset.pickle', help='specify the save file of the json file')
    parser.add_argument('--property', type=str, default='property.csv', help="file storing all the property needed to predict")
    parser.add_argument('--graph_cache', type=str, default="graph_cache.pickle", help="file storing the dict of all graph name->structure")
    args = parser.parse_args()
    print(args)

    dmin, dmax, step, var=0, 5, 0.1, 0.5
    radius, max_neighbors = 5, 8
    gdf = GaussianDistance(dmin=dmin, dmax=dmax, step=step, var=var)
    radius_dic = load_radius_dict(f'{args.data_path}/hubbard_u.yaml')
    ari = AtomCustomJSONInitializer(os.path.join(args.data_path, args.atom_feature)) # this get the embedding of the ele from atom_init.json

    print('Loading data')
    with open(f'{args.data_path}/{args.graph_cache}', 'rb') as f:
        all_graph = pickle.load(f)

    generate_pickle(path=args.data_path, save_name=args.save_path, space_file="mp_global_features.csv", property_file=args.property, graph=all_graph, ari=ari, gdf=gdf, radius_dic=radius_dic, args=args)
    