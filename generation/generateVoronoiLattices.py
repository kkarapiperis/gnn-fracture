import argparse
import numpy as np
import pandas as pd
from voronoi import Voronoi

parser.add_argument('--Nx', type=int, default=13)
parser.add_argument('--Ny', type=int, default=13)
parser.add_argument('--shape', type=str, default='hex')
parser.add_argument('--notchWidth', type=int, default=4)
parser.add_argument('--meshIdx0', type=int, default=0)
parser.add_argument('--nMeshes', type=int, default=10000)
parser.add_argument('--plot', type=int, default=0)
args = parser.parse_args()

Nx = args.Nx # Nx x Ny cells
Ny = args.Ny # Nx x Ny cells
shape = args.shape # 'hex'
notchWidth = args.notchWidth # number of missing cells in notch
meshIdx0 = args.meshIdx0 # starting mesh index (when splitting the computation in batches)
nMeshes = args.nMeshes # how many meshes to compute (when splitting the computation in batches)

# Read the lambda parameters controlling the disorder for the sampled Voronoi lattices
dataDir = '../data/' + shape +'-' + str(Nx) +'x' + str(Ny) + '-G/'
fname = dataDir + 'lamda-sampling.dat'
lamdaPairs = pd.read_csv(fname, skipinitialspace=True,skiprows=0)
lamdaPairs = lamdaPairs.iloc[:].to_numpy()
nRepeat = int(np.round(nMeshes/len(lamdaPairs)))+1
lamdaPairs = np.concatenate([lamdaPairs]*nRepeat,axis=0)
lamdaPairs = lamdaPairs[:nMeshes]

meshIdx = meshIdx0-1
for lamda in lamdaPairs:
	meshIdx += 1
	print('meshIdx = ',meshIdx)
	voronoi = Voronoi(Nx, Ny, shape, notchWidth, lamda, meshIdx)
	voronoi.sampleNuclei()
	voronoi.saveGraph(meshIdx)
	voronoi.saveDualGraph(meshIdx)
	if args.plot:
		voronoi.plotLattice()
		voronoi.plotDualLattice()