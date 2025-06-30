'''
Author: Changwei Zhang 
Date: 2023-05-23 22:38:22 
Last Modified by:   Changwei Zhang 
Last Modified time: 2023-05-23 22:38:22 
'''

from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.io.ase import AseAtomsAdaptor
import os
import numpy as np
from ase import Atoms
from numba import njit

################################ SIESTA calculation parameters Set begin #####################

basic_commad = """
SystemName      {:s}
SystemLabel     {:s}

#PAO.BasisSize           DZP
#PAO.SplitNorm           0.26

%block PS.lmax
   Ti    3
    O    3
%endblock PS.lmax


#  >>>>> KPOINTS <<<<<

%block kgrid_Monkhorst_Pack
   3  0  0  0.0
   0  3  0  0.0
   0  0  5  0.0
%endblock kgrid_Monkhorst_Pack

xc.functional      GGA    # Default vaxc.authors         HSE-06    # Default value
xc.authors         HSE06    # Default value

# >>>>> self-consistent-field stuff <<<<<

ElectronicTemperature   300. K
SCF.Mix                 Hamiltonian
SCF.Mix.First          .true.
SCF.Mix.First.Force    .false.
SCF.Mixer.Method        Pulay
SCF.Mixer.Weight        0.2
SCF.Mixer.History       6


MaxSCFIteration         1000
SCF.FreeE.Converge     .true.
SCF.FreeE.Tolerance     1.d-6 eV
SCF.DM.Converge        .true.
SCF.DM.Tolerance        1.d-4 eV
# Write.DM.end.of.cycle  .true.
# Write.H.end.of.cycle   .true.

DM.UseSaveDM           .false.
MD.UseSaveCG           .false.

SaveHS                 .true.
SaveRho                .true.
# OR COOP.Write .true.

Mesh.Cutoff             300. Ry     # Reduce eggbox effect


# DM.UseSaveDM            T
# MeshCutoff              250. Ry     # Equivalent planewave cutoff for the grid
# MaxSCFIterations        50         # Maximum number of SCF iterations per step
# DM.MixingWeight         0.1         # New DM amount for next SCF cycle
# DM.Tolerance            1.d-4       # Tolerance in maximum difference
#                                     # between input and output DM
# DM.NumberPulay          6          # Number of SCF steps between pulay mixing

# >>>>> Eigenvalue problem: order-N or diagonalization <<<<<

SolutionMethod          diagon      # OrderN or Diagon

MD.TypeOfRun            CG          # Type of dynamics:
MD.Steps                0
MD.VariableCell        .false.
MD.MaxForceTol          0.01 eV/Ang
MD.MaxDispl             0.15 Bohr

# MD.VariableCell        .true.
# MD.MaxForceTol          0.01 eV/Ang
# #WriteMDXmol           .true.
# WriteForces            .true.
# WriteCoorStep

HFX.MinimumNumberGaussians        3
HFX.MaximumNumberGaussians        6
HFX.UseFittedNAOs                 T
HFX.TruncateDM                    T
HFX.FarField                      T
HFX.FarFieldTolerance     0.100E-05
HFX.PairListTolerance     0.100E-05
HFX.SchwarzTolerance      0.100E-05
HFX.StoreERIsTolerance    0.100E-05
HFX.GaussianEPS             1.0E-05
HFX.Dynamic_parallel              F
HFX.FragSize                  50000


NetCharge  2

\n"""

@njit
def check_bound(poses:np.ndarray, cell:np.ndarray):
    invcell = np.linalg.inv(cell)
    direct = np.zeros_like(poses)
    for i, pos in enumerate(poses):
        dir = pos @ invcell
        direct[i] = dir
    showmessage = not(np.all(direct < 1) and np.all(direct >= 0))
    direct = direct % 1
    for i, dir in enumerate(direct):
        pos = dir @ cell
        poses[i] = pos
    return poses, showmessage

def ase_atoms_to_siestafile(atoms:Atoms, basic_commad:str, filename:str):
    chemical_symbols = atoms.get_chemical_symbols()
    species = list(set(chemical_symbols))
    positions = atoms.get_array(name='positions')
    cell = atoms.get_cell().array
    positions, showmessage = check_bound(positions, cell)
    siesta = basic_commad
    siesta += "#\n# Definition of Atomic Species\n#\n"
    siesta += f'NumberOfSpecies       {len(species)}\n'
    siesta += '%block ChemicalSpeciesLabel\n'
    for idx, s in enumerate(species):
        siesta += f"  {idx+1}  {Element(s).Z}  {s}\n"    
    siesta += "%endblock ChemicalSpeciesLabel\n\n"

    siesta += "%block PAO.Basis\n"
    siesta += """
Ti    5      1.9243229
 n=3    0    1   E     89.2544944       2.4608930
   2.8872099
   1.00000000000000
 n=4    0    2   E     97.7980549       4.71452135
   4.8057173            4.6151393
   1.00000000000000        1.00000000000000
 n=3    1    1   E     73.5285737       2.7406764
   3.1432229
   1.00000000000000
 n=4    1    1   E     5.4801522       2.0394443
   3.0013932
   1.00000000000000
 n=3    2    2   E     28.9714917       3.4223389
    4.0842297            2.6826474
   1.00000000000000        1.00000000000000
O     3     -.2126369
 n=2    0    2   E      27.7694723       3.5781202
   4.3372307            2.6101394
   1.00000000000000        1.00000000000000
 n=2    1    2   E      25.8458591       4.1751665
   4.8459167            2.8438221
   1.00000000000000        1.00000000000000
 n=3    2    1   E      2.4082762       1.3604859
   3.3849443
   1.00000000000000
                """
    siesta += "\n%endblock PAO.Basis\n"

    siesta += "#\n# Atoms\n#\n"
    siesta += f"NumberOfAtoms         {len(chemical_symbols)}\n\n"
    siesta += "AtomicCoordinatesFormat   Ang # Ang|Bohr|Fractional\n"
    siesta += "%block AtomicCoordinatesAndAtomicSpecies\n"
    for num, sym in enumerate(chemical_symbols):
        siesta += "  %10.7f  %10.7f  %10.7f   %d\n" % (*positions[num], species.index(sym)+1)
    siesta += "%endblock AtomicCoordinatesAndAtomicSpecies\n\n"
    siesta += "LatticeConstant      1.00 Ang\n"
    siesta += "%block LatticeVectors\n"
    siesta += "      %10.7f  %10.7f  %10.7f\n      %10.7f  %10.7f  %10.7f\n      %10.7f  %10.7f  %10.7f\n" % (*cell[0], *cell[1], *cell[2])
    siesta += "%endblock LatticeVectors\n"

    siesta +="\n%block NAO2GTO\n"
    siesta += """
Ti 7
3 0 1 5
5.06304789829199            -35.6717924214391
3.31373807719686            76.8633303931056 
0.670406644152027           1.72872108681444 
3.71496347430559            -154.24549388317 
4.41598554030901            112.587194855657 
4 0 1 6
0.441202974268689            -150.071379747411
5.27690582705904             103.720408211751 
0.524649781222156            53.3428565589496 
0.370573053550749            175.428804717205 
0.326156097278806            -77.4652778840466
5.26166516355046             -104.224964847292
4 0 2 5                    
0.694500200728577            43.2802558385826  
0.482936642641355            273.643267496283  
0.554233890979043            -194.211444396034 
0.431394162021397            -121.065711561792 
3.10435960723413             -0.549188284035539
3 1 1 3
0.632915630171411           1.03009857098527
2.11444558310662            7.79466290595463
2.46174717371221            -5.4840915250327
4 1 1 5
2.44863530566875             10.2838086668641 
0.554644500179955            4.83590130683068 
0.285750541769038            13.759214643676  
2.74065831252238             -7.62308347297908
0.31190182456528             -17.9366481463711
3 2 1 4
1.36870631350254             1.98082982444185 
8.01937397877435             -1.39545231841736
0.409711034427782            0.299331504076561
4.10675115902018             8.26162660660348 
3 2 2 4
5.50433570405124            23.1671423976058 
0.951875520264245           1.66355875198474 
2.70880774765734            4.15011215011257 
6.13700886686494            -17.6813717082228
O  5
2 0 1 5
3.41980229494729             38.7112071561663 
0.277547511404403            0.498414181114465
0.810602030035813            1.1046965794338  
3.99862230308868             37.786767450746  
3.70364631037674             -76.8500050844386
2 0 2 5
3.25363978744146             34.3761166857969 
4.02608819522255             32.1247330919085 
3.6300647458624              -66.8119177811748
0.490954126840092            -11.7157033807992
0.51254347849647             13.4742642380047 
2 1 1 4
1.0970477510991              1.53037064233019  
0.268550033357923            0.294903399746936 
9.50887903084867             -0.530849047711627
4.00768745526609             4.01385819243416  
2 1 2 4
6.08326226721387            -144.645028465317
2.29021821038278            2.18740789111945 
6.02843270909986            147.665876649304 
0.772755176743678           1.44934972343959 
3 2 1 4
2.32296489876456            0.634437511263633  
0.468018655042551           -14.3421215097284  
0.464760624115432           14.8356099342484   
18.6950379551616            -0.0135576634197983
    """
    siesta +="\n%endblock NAO2GTO"
    with open(filename,'w') as wf:
        wf.write(siesta)

################################ SIESTA calculation parameters Set end #######################
from glob import glob
if __name__ == '__main__':
    ######################## Input parameters begin #######################
    system_name = 'TiO2'
    #poscar_path = ["./input/{:d}.vasp".format(i) for i in range(1,501)] # The path of poscar or cif files
    vasp_path = './structure'
    vasp_files = glob(os.path.join(vasp_path, '*.vasp'))
    filepath = './input_siesta' # siesta file directory to save
    
    ######################## Input parameters end #########################

    if not os.path.exists(filepath):
        os.mkdir(filepath)

    for i, poscar in enumerate(vasp_files):
        cif_id = str(i+1)

        crystal = Structure.from_file(poscar)
        ase_atoms = AseAtomsAdaptor.get_atoms(crystal)
        cell = ase_atoms.get_cell().array
        filename =  os.path.join(filepath, f'{system_name}_'+ cif_id + ".fdf")
        ase_atoms_to_siestafile(ase_atoms, 
                                basic_commad.format(system_name, system_name), # or ''
                                filename)

