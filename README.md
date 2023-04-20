# gnn-fracture
Prediction and control of fracture paths in disordered architected materials using graph neural networks

## Pipeline
To generate the Voronoi architectures run: generateVoronoiLattices.py

- **generation/generateVoronoiLattices.py** generates the Voronoi lattices and their corresponding meshes, whose mechanical response is evaluated using the open-source finite element code ae108 [link](https://www.ae108.ethz.ch).
- 
-   used presented framework with lattice-stiffness pairs computed using an inhouse finite element simulation. **It should only be used if one is interested to retrain the networks**, e.g., to reproduce the evaluation presented in the publication. The training dataset can be found under this [link](https://www.research-collection.ethz.ch/handle/20.500.11850/520254).
- **optimize.py** predicts and stores a variety of inverse designs given a certain set of anistropic stiffness tensors. These tensors must be provided in Voigt notation via a .csv-file in 'data/prediction.csv', as, e.g., the provided anisotropic bone samples which can be run to reproduce the presented results. (Note that the predicted stiffnesses slightly differ from the ones presented in the publication, which were computed and verified using our inhouse finite element framework.)
- **main_export.py** should only be executed after main_predict.py. It plots the predicted lattice descriptor and converts it into a list of nodal position and connectivities for further postprocessing. Additionally, it warns the user if the requested stiffness is to stiff or soft for the considered range of relative densities and Young's modulus, in which case main_predict.py should be rerun with a suitable Young's modulus of the base material.

## Requirements

- Python (tested on version 3.9.1)
- Python packages:
  - Pandas
  - Numpy
  - Scipy
  - NetworkX
  - Pytorch 
  - Matplotlib

