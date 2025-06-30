'''
Descripttion: Script for converting poscar to openmx input file
version: 0.1
Author: Yang Zhong
Date: 2022-11-24 19:03:36
LastEditors: Yang Zhong
LastEditTime: 2022-11-24 19:06:33
'''

import json
from pymatgen.core.structure import Structure
import glob
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import ase
from ase import Atoms
import numpy as np
import os
from tqdm import tqdm
import natsort
import random

def ase_atoms_to_openmxfile(atoms:Atoms, basic_commad:str, spin_set:dict, PAO_dict:dict, PBE_dict:dict, filename:str):
    chemical_symbols = atoms.get_chemical_symbols()
    species = set(chemical_symbols)
    positions = atoms.get_array(name='positions')
    cell = atoms.get_cell().array
    openmx = basic_commad
    openmx += "#\n# Definition of Atomic Species\n#\n"
    openmx += f'Species.Number       {len(species)}\n'
    openmx += '<Definition.of.Atomic.Species\n'
    for s in species:
        openmx += f"{s}   {PAO_dict[s]}       {PBE_dict[s]}\n"    
    openmx += "Definition.of.Atomic.Species>\n\n"
    openmx += "#\n# Atoms\n#\n"
    openmx += "Atoms.Number%12d" % len(chemical_symbols)
    openmx += "\nAtoms.SpeciesAndCoordinates.Unit   Ang # Ang|AU"
    openmx += "\n<Atoms.SpeciesAndCoordinates           # Unit=Ang."
    for num, sym in enumerate(chemical_symbols):
        openmx += "\n%3d  %s  %10.7f  %10.7f  %10.7f   %.2f   %.2f" % (num+1, sym, *positions[num], *spin_set[chemical_symbols[num]])
    openmx += "\nAtoms.SpeciesAndCoordinates>"
    openmx += "\nAtoms.UnitVectors.Unit             Ang #  Ang|AU"
    openmx += "\n<Atoms.UnitVectors                     # unit=Ang."
    openmx += "\n      %10.7f  %10.7f  %10.7f\n      %10.7f  %10.7f  %10.7f\n      %10.7f  %10.7f  %10.7f" % (*cell[0], *cell[1], *cell[2])
    openmx += "\nAtoms.UnitVectors>"
    with open(filename,'w') as wf:
        wf.write(openmx)

basic_commad = """#
#      File Name      
#

System.CurrrentDirectory         ./    # default=./
System.Name                      GaAs
DATA.PATH           /data/home/yxma/DFT_DATA19   # default=../DFT_DATA19
level.of.stdout                   1    # default=1 (1-3)
level.of.fileout                  1    # default=1 (0-2)
HS.fileout                   on       # on|off, default=off

#
# SCF or Electronic System
#

scf.XcType                  GGA-PBE    # LDA|LSDA-CA|LSDA-PW|GGA-PBE
scf.SpinPolarization        off        # On|Off|NC
scf.ElectronicTemperature  300.0       # default=300 (K)
scf.energycutoff           300.0       # default=150 (Ry)
scf.maxIter                 200         # default=40
scf.EigenvalueSolver       Band      # DC|GDC|Cluster|Band
scf.Kgrid                  3 3 3       # means 4x4x4
scf.Mixing.Type           Rmm-Diisk     # Simple|Rmm-Diis|Gr-Pulay|Kerker|Rmm-Diisk
scf.Init.Mixing.Weight     0.10        # default=0.30 
scf.Min.Mixing.Weight      0.001       # default=0.001 
scf.Max.Mixing.Weight      0.400       # default=0.40 
scf.Mixing.History          7          # default=5
scf.Mixing.StartPulay       5          # default=6
scf.criterion             1.0e-7      # default=1.0e-6 (Hartree) 
scf.system.charge           0
#
# MD or Geometry Optimization
#

MD.Type                      Nomd        # Nomd|Opt|NVE|NVT_VS|NVT_NH
                                       # Constraint_Opt|DIIS2|Constraint_DIIS2
MD.Opt.DIIS.History          4
MD.Opt.StartDIIS             5         # default=5
MD.maxIter                 100         # default=1
MD.TimeStep                1.0         # default=0.5 (fs)
MD.Opt.criterion          1.0e-4       # default=1.0e-4 (Hartree/bohr)

#
# MO output
#

MO.fileout                  off        # on|off, default=off
num.HOMOs                    2         # default=1
num.LUMOs                    2         # default=1

#
# DOS and PDOS
#

Dos.fileout                  off       # on|off, default=off
Dos.Erange              -10.0  10.0    # default = -20 20 
Dos.Kgrid                 1  1  1      # default = Kgrid1 Kgrid2 Kgrid3

\n"""

spin_set = {'H':[0.5, 0.5],
            'He':[1.0,1.0],
            'Li':[1.5,1.5],
            'Be':[1.0,1.0],
            'B':[1.5,1.5], 
            'C':[2.0, 2.0], 
            'N': [2.5,2.5], 
            'O':[3.0,3.0], 
            'F':[3.5,3.5],
            'Ne':[4.0,4.0],
            'Na':[4.5,4.5],
            'Mg':[4.0,4.0],
            'Al':[1.5,1.5],
            'Si':[2.0,2.0],
            'P':[2.5,2.5],
            'S':[3.0,3.0],
            'Cl':[3.5,3.5],
            'Ar':[4.0,4.0],
            'K':[4.5,4.5],
            'Ca':[5.0,5.0],
            'Si': [2.0, 2.0], 
            'Mo': [7.0, 7.0], 
            'S': [3.0, 3.0],
            'Bi':[7.5, 7.5],
            'Se':[3.0, 3.0],
            'Ga':[6.5,6.5],
            'As':[7.5,7.5]
            }

PAO_dict = {'H':'H6.0-s2p1',
            'He':'He8.0-s2p1',
            'Li':'Li8.0-s3p2',
            'Be':'Be7.0-s2p2',
            'B':'B7.0-s2p2d1',
            'C':'C6.0-s2p2d1',
            'N':'N6.0-s2p2d1',
            'O':'O6.0-s2p2d1',
            'F':'F6.0-s2p2d1',
            'Ne':'Ne9.0-s2p2d1',
            'Na':'Na9.0-s3p2d1',
            'Mg':'Mg9.0-s3p2d1',
            'Al':'Al7.0-s2p2d1',
            'Si':'Si7.0-s2p2d1',
            'P':'P7.0-s2p2d1',
            'S':'S7.0-s2p2d1',
            'Cl':'Cl7.0-s2p2d1',
            'Ar':'Ar9.0-s2p2d1',
            'K':'K10.0-s3p2d1',
            'Ca':'Ca9.0-s3p2d1',
            'Bi':'Bi8.0-s3p2d2',
            'Se':'Se7.0-s3p2d2',
            'Ga':'Ga7.0-s3p2d2',
            'As':'As7.0-s3p2d2'
            }

PBE_dict = {'H':'H_PBE19',
            'He':'He_PBE19',
            'Li':'Li_PBE19',
            'Be':'Be_PBE19',
            'B':'B_PBE19',
            'C':'C_PBE19',
            'N':'N_PBE19',
            'O':'O_PBE19',
            'F':'F_PBE19',
            'Ne':'Ne_PBE19',
            'Na':'Na_PBE19',
            'Mg':'Mg_PBE19',
            'Al':'Al_PBE19',
            'Si':'Si_PBE19',
            'P':'P_PBE19',
            'S':'S_PBE19',
            'Cl':'Cl_PBE19',
            'Ar':'Ar_PBE19',
            'K':'K_PBE19',
            'Ca':'Ca_PBE19',
            'Bi':'Bi_PBE19',
            'Se':'Se_PBE19',
            'Ga':'Ga_PBE19',
            'As':'As_PBE19'
            }

filepath = './input' # openmx input file directory
if not os.path.exists(filepath):
    os.mkdir(filepath)

f_vasp = glob.glob("./structure/*.vasp") # poscar or cif file directory
f_vasp = natsort.natsorted(f_vasp)

for i, poscar in enumerate(f_vasp):
    cif_id = str(i+1)
    crystal = Structure.from_file(poscar)
    ase_atoms = AseAtomsAdaptor.get_atoms(crystal)
    cell = ase_atoms.get_cell().array
    filename =  os.path.join(filepath, 'GaAs_' + cif_id + ".dat")
    ase_atoms_to_openmxfile(ase_atoms, basic_commad, spin_set, PAO_dict, PBE_dict, filename)

