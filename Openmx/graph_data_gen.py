'''
Descripttion: The scripts used to generate the input file graph_data.npz for HamNet.
version: 0.1
Author: Yang Zhong
Date: 2022-11-24 19:07:54
LastEditors: Yang Zhong
LastEditTime: 2023-03-11 16:47:06
'''

import json
import numpy as np
import os
import sys
from torch_geometric.data import Data
import torch
import glob
import natsort
from tqdm import tqdm
import re
from pymatgen.core.periodic_table import Element

basis_def = {1:np.array([0,1,3,4,5], dtype=int), # H
             2:np.array([0,1,3,4,5], dtype=int), # He
             3:np.array([0,1,2,3,4,5,6,7,8], dtype=int), # Li
             4:np.array([0,1,3,4,5,6,7,8], dtype=int), # Be
             5:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # B
             6:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # C
             7:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # N
             8:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # O
             9:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # F
             10:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Ne
             11:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Na
             12:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Mg
             13:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Al
             14:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Si
             15:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # p
             16:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # S
             17:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Cl
             18:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Ar
             19:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # K
             20:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Ca 
             42:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], dtype=int), # Mo  
             83:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], dtype=int), # Bi  
             34:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], dtype=int), # Se 
             24:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Cr 
             53:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], dtype=int), # I   
             82:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], dtype=int), # pb
             55:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], dtype=int), # Cs
	     31:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], dtype=int), # Ga
	     33:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], dtype=int)  # As
 	
             }
doping_charge = 0
nao_max = 19
au2ang = 0.5291772083
pattern_eng = re.compile(r'Enpy  =(\W+)(\-\d+\.?\d*)')
pattern_md = re.compile(r'MD= 1  SCF=(\W*)(\d+)')
pattern_latt = re.compile(r'<Atoms.UnitVectors.+\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+Atoms.UnitVectors>')
pattern_coor = re.compile(r'\s+\d+\s+(\w+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+\-?\d+\.?\d+\s+\-?\d+\.?\d+')

graphs = dict()
graph_data_path = '/data/home/yxma/workplace2/openmx_charge0/calc'
if not os.path.exists(graph_data_path):
    os.makedirs(graph_data_path)
read_openmx_path = '/data/home/yxma/software/openmx/openmx_tools/'
read_openmx_path = os.path.join(read_openmx_path, 'read_openmx')
#print(read_openmx_path)
system_name = 'GaAs'
f_scfout = glob.glob("/data/home/yxma/workplace2/openmx_charge0/calc/data/GaAs-*/GaAs.scfout")
f_scfout = natsort.natsorted(f_scfout)
#print(f_scfout)
for idx, f_sc in enumerate(tqdm(f_scfout)):
    print(f_sc)
    # read energy
    f_std = f_sc[:-(7+len(system_name))] + "log.out"
    f_dat = f_sc[:-(7+(len('GaAs')))] + 'GaAs.dat'
    f_H0 = f_sc[:-(7+len(system_name))] + "overlap.scfout"
    #print(f_std)
    #print(f_dat)
    #print(f_H0)
    try:
        with open(f_std, 'r') as f:
            content = f.read()
            #print(content)
            Enpy = float(pattern_eng.findall((content).strip())[0][-1])
            max_SCF = int(pattern_md.findall((content).strip())[-1][-1])
    except:
        #print('error')
        #continue
        Enpy = 0.0
        max_SCF = 1    
    # skip condition 1
    if max_SCF > 188:
        continue  
    
    # Read crystal parameters
    try:
        with open(f_dat,'r') as f:
            content = f.read()
            #print(content)
            speciesAndCoordinates = pattern_coor.findall((content).strip())
            latt = pattern_latt.findall((content).strip())[0]
            latt = np.array([float(var) for var in latt]).reshape(-1, 3)/au2ang
    
            species = []
            coordinates = []
            for item in speciesAndCoordinates:
                species.append(item[0])
                coordinates += item[1:]
            z = atomic_numbers = np.array([Element[s].Z for s in species])
            coordinates = np.array([float(pos) for pos in coordinates]).reshape(-1, 3)/au2ang
    except:
        continue
    
    # skip condition 2
    if len(z) > 600:
        continue
    
    # read hopping parameters
    os.system(read_openmx_path + " " + f_sc)
    if not os.path.exists("./HS.json"):
        continue
    
    with open("./HS.json",'r') as load_f:
        load_dict = json.load(load_f)
        #print(load_dict)
        pos = np.array(load_dict['pos'])
        edge_index = np.array(load_dict['edge_index'])
        inv_edge_idx = np.array(load_dict['inv_edge_idx'])
        #
        Hon = load_dict['Hon'][0]
        Hoff = load_dict['Hoff'][0]
        Son = load_dict['Son']
        Soff = load_dict['Soff']
        nbr_shift = np.array(load_dict['nbr_shift'])
        cell_shift = np.array(load_dict['cell_shift'])
        
        # Find inverse edge_index
        if len(inv_edge_idx) != len(edge_index[0]):
            print('Wrong info: len(inv_edge_idx) != len(edge_index[0]) !')
            sys.exit()

        #
        num_sub_matrix = pos.shape[0] + edge_index.shape[1]
        H = np.zeros((num_sub_matrix, nao_max**2))
        S = np.zeros((num_sub_matrix, nao_max**2))
        
        for i, (sub_maxtrix_H, sub_maxtrix_S) in enumerate(zip(Hon, Son)):
            mask = np.zeros((nao_max, nao_max), dtype=int)
            src = z[i]
            mask[basis_def[src][:,None], basis_def[src][None,:]] = 1
            mask = (mask > 0).reshape(-1)
            H[i][mask] = np.array(sub_maxtrix_H)
            S[i][mask] = np.array(sub_maxtrix_S)
        
        num = 0
        for i, (sub_maxtrix_H, sub_maxtrix_S) in enumerate(zip(Hoff, Soff)):
            mask = np.zeros((nao_max, nao_max), dtype=int)
            src, tar = z[edge_index[0,num]], z[edge_index[1,num]]
            mask[basis_def[src][:,None], basis_def[tar][None,:]] = 1
            mask = (mask > 0).reshape(-1)
            H[num + len(z)][mask] = np.array(sub_maxtrix_H)
            S[num + len(z)][mask] = np.array(sub_maxtrix_S)
            num = num + 1
    os.system("rm HS.json")
    
    # read H0
    os.system(read_openmx_path + " " + f_H0)
    if not os.path.exists("./HS.json"):
        continue
    
    with open("./HS.json",'r') as load_f:
        load_dict = json.load(load_f)
        Hon0 = load_dict['Hon'][0]
        Hoff0 = load_dict['Hoff'][0]

        #
        num_sub_matrix = pos.shape[0] + edge_index.shape[1]
        H0 = np.zeros((num_sub_matrix, nao_max**2))
        
        for i, sub_maxtrix_H in enumerate(Hon0):
            mask = np.zeros((nao_max, nao_max), dtype=int)
            src = z[i]
            mask[basis_def[src][:,None], basis_def[src][None,:]] = 1
            mask = (mask > 0).reshape(-1)
            H0[i][mask] = np.array(sub_maxtrix_H)
        
        num = 0
        for i, sub_maxtrix_H in enumerate(Hoff0):
            mask = np.zeros((nao_max, nao_max), dtype=int)
            src, tar = z[edge_index[0,num]], z[edge_index[1,num]]
            mask[basis_def[src][:,None], basis_def[tar][None,:]] = 1
            mask = (mask > 0).reshape(-1)
            H0[num + len(z)][mask] = np.array(sub_maxtrix_H)
            num = num + 1
    os.system("rm HS.json")
    
    # save in Data
    graphs[idx] = Data(z=torch.LongTensor(z),
                        cell = torch.Tensor(latt[None,:,:]),
                        total_energy=Enpy,
                        pos=torch.FloatTensor(pos),
                        node_counts=torch.LongTensor([len(z)]),
                        edge_index=torch.LongTensor(edge_index),
                        inv_edge_idx=torch.LongTensor(inv_edge_idx),
                        nbr_shift=torch.FloatTensor(nbr_shift),
                        cell_shift=torch.LongTensor(cell_shift),
                        hamiltonian=torch.FloatTensor(H),
                        overlap=torch.FloatTensor(S),
                        Hon = torch.FloatTensor(H[:pos.shape[0],:]),
                        Hoff = torch.FloatTensor(H[pos.shape[0]:,:]),
                        Hon0 = torch.FloatTensor(H0[:pos.shape[0],:]),
                        Hoff0 = torch.FloatTensor(H0[pos.shape[0]:,:]),
                        Son = torch.FloatTensor(S[:pos.shape[0],:]),
                        Soff = torch.FloatTensor(S[pos.shape[0]:,:]),
                        doping_charge = torch.FloatTensor([doping_charge]))

graph_data_path = os.path.join(graph_data_path, 'graph_data_follow.npz')
np.savez(graph_data_path, graph=graphs)
