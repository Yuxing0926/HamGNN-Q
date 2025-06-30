# HamGNN-Q: Predicting Charged Defect Hamiltonians with Equivariant Graph Neural Networks

**HamGNN-Q** is a graph neural network tool for predicting DFT-quality Hamiltonian matrices of materials systems — including both **perfect** and **defective structures**, with arbitrary charge states — without requiring quantum mechanical calculations.

## What It Does
- Predicts Hamiltonian Matrices directly from atomic structures and background charges
- Supports charged or neutral systems, including point defects, vacancies, interstitials, or perfect crystals
- Handles varied charge states through a background charge embedding
- Enables downstream tasks such as:
  - Polaron wavefunction estimation
  - Electronic property analysis
  - Defect state localization studies

## Directory Overview
- `/HamGNN/`  
  The core implementation of the HamGNN-Q model. You can run the training or prediction pipeline by executing:
  ```bash
  python HamGNN/main.py
- `/Openmx/` and `Honpas`
  - Converting structures from POSCAR format to OpenMX(`.dat`) and Honpas (`.fdf`) input formats
  - Constructing graph representations from OpenMX or Honpas output files

## Dataset building
1. **Generate Structure Files:** Create structure files via molecular dynamics or random perturbation.
2. **Self-Consistent Calculation:** Perform self-consistent field (SCF) calculations to obtain the converged Hamiltonian matrix corresponding to the relaxed charge density.
3. **Processing with postprocess script:** Generate the overlap file, which contains the Hamiltonian matrix H0, independent of the self-consistent charge density.
4. **Graph generation:** Use `graph_data_gen.py` to convert the structure files into graph representations suitable for input into the graph neural network.

We provide example calculation setups for charged defect systems using three different DFT software packages:
### OpenMX
1. **Convert to Openmx Format:** Convert the structure to be predicted into `.dat` format using poscar2openmx.py.
2. **Run HONPAS:** Generate the overlap data (`overlap.scfout`) for the structure.
3. **Generate Graph Data:** Use the `predict_data_gen.py` script to package the data into a `graph_data.npz` file for training or prediction.
   
### Honpas
1. **Convert to Siesta Format:** Convert the structure to be predicted into `.fdf` format using poscar2siesta.py.
2. **Run HONPAS:** Generate the overlap data (`overlap.HSX`) for the structure.
3. **Generate Graph Data:** Use the `predict_data_gen_siesta.py` script to package the data into a `graph_data.npz` file for training or prediction.

### ABACUS

## References
1. Y. Ma, et al.Transferable machine learning approach for predicting electronic structures of charged defects. (Feature Article) [https://doi.org/10.1063/5.0242683](https://doi.org/10.1063/5.0242683)

## Code contributors:
+ Yuxing Ma (Fudan University)
+ Yang Zhong (Fudan University)




