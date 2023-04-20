import os
import sys
import time
import torch
import random
import itertools
import numpy as np
import pandas as pd
from torch_geometric.loader import DataListLoader
from torch_geometric.data import Data, DataLoader, Batch

def splitTrainTestMeshes(Nx, Ny, shape, Ndata, minDisorder, maxDisorder, test_size, seed=0):
	"""
	Return indices of meshes used for training/testing
	"""
	# Directories
	rootDir = '../data/'
	dataDirG = rootDir + str(shape) + '-' + str(Nx) + 'x' + str(Ny) + '-G/'

	# Read geometric disorder for each mesh
	H = pd.read_csv(dataDirG + 'mesh-list.dat', skipinitialspace=True,  skiprows=0)
	HGeom = H.iloc[:].to_numpy()[:,3]
	HGeom = HGeom[:Ndata]
	HGeom_mask = (HGeom>=minDisorder) & (HGeom<=maxDisorder)
	mesh_list = np.arange(Ndata)[HGeom_mask] 

	# Shuffle the meshes and split to test, train
	np.random.seed(seed)
	Ndata = len(mesh_list)
	np.random.shuffle(mesh_list)
	mesh_split = int(np.floor(test_size * Ndata)) 
	train_mesh_idx = mesh_list[mesh_split:]
	test_mesh_idx = mesh_list[:mesh_split]

	return train_mesh_idx, test_mesh_idx

def readNormalization(Nx, Ny, shape):
	"""
	Read normalization constants 
	"""
	# Directories
	rootDir = '../data/'
	dataDirGDual = rootDir + str(shape) + '-' + str(Nx) + 'x' + str(Ny) + '-G-dual/'
	x_bounds = np.loadtxt(dataDirGDual + 'x_normalization.dat')
	r_bounds = np.loadtxt(dataDirGDual + 'r_normalization.dat')
	return x_bounds, r_bounds

def getMaxSequenceLength(Nx, Ny, shape, train_mesh_idx):

	# Initialize 
	maxSequenceLength = 0

	# Data dir
	rootDir = '../data/'
	dataDirTFrac = rootDir + str(shape) + '-' + str(Nx) + 'x' + str(Ny) + '-T-frac/'

	for meshIdx in train_mesh_idx:

		# Read broken edge sequence
		fname = dataDirTFrac + 'mesh-' + str(meshIdx) + '-broken-edges-ordered.dat'
		broken_edges = np.loadtxt(fname).astype(int)
		maxSequenceLength = max(maxSequenceLength, len(broken_edges)-2)

	return maxSequenceLength

def getDataLoaders(Nx, Ny, shape, notchWidth, meshList, normalize, augment, maxSequenceLength, batch_size, shuffle):
	"""
	Read data from files and return torch data loader
	"""
	# Directories
	rootDir = '../data/'
	dataDirGDual = rootDir + str(shape) + '-' + str(Nx) + 'x' + str(Ny) + '-G-dual/'
	dataDirTFrac = rootDir + str(shape) + '-' + str(Nx) + 'x' + str(Ny) + '-T-frac/'
	dataDirG = rootDir + str(shape) + '-' + str(Nx) + 'x' + str(Ny) + '-G/'

	# Read normalizing constants 
	if normalize:
		X_bounds = np.loadtxt(dataDirGDual + 'x_normalization.dat')
		r_bounds = np.loadtxt(dataDirGDual + 'r_normalization.dat')
		X_min,X_max = X_bounds
		r_x_min,r_x_max = r_bounds[0]
		r_y_min,r_y_max = r_bounds[1]
		r_l_min,r_l_max = r_bounds[2]
		
	# Initialize torch 'Data' list
	data_list = []
	mesh_list = []

	if augment:
		mirMap, midY = findNodeMappingForDataAugmentation(Nx,Ny)

	for meshIdx in meshList:

		# Filter outliers with very long fracture sequences
		fname = dataDirTFrac + 'mesh-' + str(meshIdx) + '-broken-edges-ordered.dat'
		broken_edges = np.loadtxt(fname).astype(int)
		if len(broken_edges.shape) != 2: continue
		if len(broken_edges) > maxSequenceLength: continue

		#=====================
		# Read dual mesh
		#=====================
		meshStr = dataDirGDual + 'mesh-' + str(meshIdx) + '-dual.dat'
		nodes = []
		edges = []
		nodeOrEdge = 'node'
		with open(meshStr) as f:
			for line in f:
				line = line.rstrip('\n')
				if line=='connectivity':
					nodeOrEdge = 'edge'
				else:
					if nodeOrEdge == 'node':
						nodes.append([float(n) for n in line.split(',')])
					else:
						edges.append([int(n) for n in line.split(',')])

		nodes = np.array(nodes)
		edges = np.array(edges)
		num_nodes = len(nodes)

		# Convert to adjacency matrix
		A = np.zeros((num_nodes,num_nodes))
		for e in edges:
			A[e[0]][e[1]] = 1
			A[e[1]][e[0]] = 1

		#=====================
		# Read nodal features and perform some transformations
		#=====================
		meshXStr = dataDirGDual + 'mesh-' + str(meshIdx) + '-X.dat'
		X = np.loadtxt(meshXStr, delimiter=',')

		# Do some modifications on the features to obtain:
		# [Volume, magnitude of anisotropy, major axis orientation, 
		# circularity, min edge length, Z, order parameter]
		XMagAnisotr = np.linalg.norm(X[:,1:3],axis=1)
		# Make sure major axis is pointing towards +x when computing theta
		XMajAxis = np.einsum('ij,i->ij',X[:,3:5], np.sign(X[:,3]+1e-6))
		XMajAxisTheta = np.arctan2(XMajAxis[:,1],XMajAxis[:,0])
		# Order parameter as geometric clustering feature (Janssens et al, 2006)
		op = np.zeros(num_nodes)
		for i in range(num_nodes):
			for j in range(num_nodes):
				if A[i][j] == 1:
					op[i] += np.linalg.norm(nodes[i]-nodes[j]) 
		# Catch any problematic data
		if np.any(np.absolute(X[:,0]) < 1e-6) or np.any(np.absolute(X[:,-2]) < 1e-6): 
			continue
		if np.any(np.absolute(X[:,-3]) > 100): 
			continue
		X = np.c_[1/X[:,0], XMagAnisotr, XMajAxisTheta, X[:,-3], 1/X[:,-2], X[:,-1],op]
		X = np.hstack((nodes,X))

		#=====================
		# Read edge features
		#=====================
		#(relative position X,Y, primal edge length)
		meshEStr = dataDirGDual + 'mesh-' + str(meshIdx) + '-E.dat'
		r = np.loadtxt(meshEStr, delimiter=',')
		# Switch to magnitude and orientation of the relative position of connecting nodes
		relPos = np.einsum('ij,i->ij',r[:,:2], np.sign(r[:,0]+1e-6))
		r[:,0] = np.arctan2(relPos[:,1],relPos[:,0])
		r[:,1] = np.linalg.norm(relPos,axis=1)

		# Find notch nodes and define a helpful function for later
		notch_nodes = []
		node_idx = 0
		for j in range(Ny):
			for i in range(Nx-(j+1)%2):
				if j == (Ny-1)/2 and i < notchWidth: 
					notch_nodes.append(node_idx)
				node_idx += 1

		# To start off the sequence, add the existing notch edges as broken edges
		notch_edges = []
		for idxi,i in enumerate(notch_nodes):
			for j in notch_nodes[idxi:]:
				if A[i][j] > 0:
					notch_edges.append([i,j])

		for notch_edge in notch_edges[::-1]:
			broken_edges = np.insert(broken_edges,0,notch_edge,axis=0)

		# Adjust A
		idx_nonzero_A0 = np.array(np.nonzero(A)).T
		A[broken_edges[:len(notch_edges),0], broken_edges[:len(notch_edges),1]] = 0 
		A[broken_edges[:len(notch_edges),1], broken_edges[:len(notch_edges),0]] = 0

		# Initialize list holding the steps of the sequence
		data_list_seq = []
		data_list_aug_seq = []

		for n in range(len(notch_edges),maxSequenceLength+len(notch_edges)):
			if n < len(broken_edges):

				# Find the updated adjacency matrix accounting for broken edges
				A[broken_edges[n-1,0], broken_edges[n-1,1]] = 0 
				A[broken_edges[n-1,1], broken_edges[n-1,0]] = 0
				idx_nonzero_A = np.array(np.nonzero(A)).T
				# Remove corredponding edge features
				intactEdgeIdxs = []
				for idx,e in enumerate(idx_nonzero_A0):
					if (list(e) in broken_edges[:n].tolist()): continue
					if (list(e) in broken_edges[:n,::-1].tolist()): continue
					intactEdgeIdxs.append(idx)
				rC = r[intactEdgeIdxs]

				# Adjust the Z feature (reduced around crack)
				# Also compute mask based on first n broken edges
				# Only first neighbors are active in the mask 
				XC = np.copy(X)
				M = np.zeros_like(A)
				for e in idx_nonzero_A:
					if e[0] < e[1]: continue
					check0 = e[0] in broken_edges[:n]
					check1 = e[1] in broken_edges[:n]
					if check0 and not check1:
						XC[e[1],-2] -= 1
						M[e[0],e[1]] = 1
						M[e[1],e[0]] = 1
					elif check1 and not check0:
						XC[e[0],-2] -= 1
						M[e[0],e[1]] = 1
						M[e[1],e[0]] = 1

				# The index of the next broken edge
				F = broken_edges[n]

				# Use relative crack front
				broken_nodes_prev = broken_edges[:n-1].ravel()
				if broken_edges[n-1,1] not in broken_nodes_prev:
					crackFrontPos = nodes[broken_edges[n-1,1]]
				else:
					crackFrontPos = nodes[broken_edges[n-1,0]]
				XC[:,:2] -= crackFrontPos

				# Normalize
				if normalize:
					XC = (XC-X_min)/X_max
					rC[:,0] = (rC[:,0]-r_x_min)/r_x_max
					rC[:,1] = (rC[:,1]-r_y_min)/r_y_max
					rC[:,2] = (rC[:,2]-r_l_min)/r_l_max

				# Convert to torch
				edge_index = torch.nonzero(torch.tensor(A, dtype = torch.int16))
				data_list_seq.append(Data(x = torch.tensor(XC, dtype = torch.float32), 
									  edge_index = edge_index.t().contiguous(), 
									  edge_attr = torch.tensor(rC, dtype = torch.float32), 
									  M = torch.tensor(M, dtype = torch.int16), 
									  F_idx = torch.tensor(F, dtype = torch.int16),
									  pad = 0).to('cpu')) 
				mesh_list.append(meshIdx)

				# Augment (flip vertically)
				if not augment: continue
				# Fix the mirrored adjacency matrix
				A_aug = A[:, mirMap][mirMap]

				# First mirror the node positions
				X_aug = np.copy(XC)
				ptsRelToSymAxis = X_aug[:,:2]-midY
				ptsMirrored = np.copy(ptsRelToSymAxis)
				ptsMirrored[:,1] *= -1
				ptsMirrored += midY
				ptsRelToCrackFront = X_aug[:,:2]
				ptsMirrored = np.copy(ptsRelToCrackFront)
				ptsMirrored[:,1] *= -1

				# Scalar nodal features remain the same but vectorial
				# or orientation feature quantities need mirroring
				X_aug = X_aug[mirMap]
				X_aug[:,:2] = ptsMirrored[mirMap]
				X_aug[:,4] = -X_aug[:,4] # major axis orientation
					
				# Build nested dictionary for mapping edge features
				rIdx = 0
				D = {}
				for i in range(num_nodes):
					D[i] = {}
					for j in range(num_nodes):
						if A[i][j] == 1:
							D[i][j] = rC[rIdx] 
							rIdx += 1

				# Use it to find edge features in correct mirrored order
				r_aug = []					
				idx_nonzero_A_aug = np.array(np.nonzero(A_aug)).T
				for e in idx_nonzero_A_aug:
					r_aug.append(D[mirMap[e[0]]][mirMap[e[1]]])

				# Invert theta-component of rel pos of edge 
				r_aug = np.array(r_aug)
				r_aug[:,0] = -r_aug[:,0]

				# Finally map the M, F 
				M_aug = M[:, mirMap][mirMap]
				F_aug = mirMap[F]

				# Convert to torch
				edge_index_aug = torch.nonzero(torch.tensor(A_aug, dtype = torch.int16))
				data_list_aug_seq.append(Data(x = torch.tensor(X_aug, dtype = torch.float32), 
									  edge_index = edge_index_aug.t().contiguous(), 
									  edge_attr = torch.tensor(r_aug, dtype = torch.float32), 
									  M = torch.tensor(M_aug, dtype = torch.int16), 
									  F_idx = torch.tensor(F_aug, dtype = torch.int16),
									  pad = 0).to('cpu')) 
				mesh_list.append(-meshIdx)
			else:
				# Essentially pass an empty graph but keep a fake edge for the graph conv not to complain
				A_empty = torch.zeros(A.shape, dtype = torch.int16); A_empty[0,1] = 1 
				X_empty = torch.zeros(X.shape, dtype = torch.float32)
				M_empty = torch.zeros(M.shape, dtype = torch.int16)
				r_empty = torch.tensor([[0.,0.,0.]], dtype = torch.float32)
				F_empty = torch.tensor([0,0], dtype = torch.int16) 

				edge_index = torch.nonzero(A_empty)
				data_list_seq.append(Data(x = X_empty, edge_index = edge_index.t().contiguous(), 
									  edge_attr = r_empty, M = M_empty, F_idx = F_empty, pad = 1).to('cpu')) 
				mesh_list.append(meshIdx)

				# Augment
				if not augment: continue

				edge_index = torch.nonzero(A_empty)
				data_list_aug_seq.append(Data(x = X_empty, edge_index = edge_index.t().contiguous(), 
									  edge_attr = r_empty, M = M_empty, F_idx = F_empty, pad = 1).to('cpu')) 
				mesh_list.append(-meshIdx)

		# Append whole sequence to the overall data list
		data_list.append(data_list_seq)
		if augment:
			data_list.append(data_list_aug_seq)

	# Remove data from end of list to make it a multiple of the batch size
	n_remove = len(data_list) % batch_size
	for n in range(n_remove):
		data_list.pop()

	# Shuffle the overall data_list (i.e. shuffle the meshes)
	if shuffle:
		random.shuffle(data_list)
	n_meshes = len(data_list)

	# Populate list of loaders for each step
	loaders = []
	for step in range(maxSequenceLength):
		loaders.append(DataLoader([data_list[mesh_idx][step] for mesh_idx in range(n_meshes)], 
			batch_size=batch_size, shuffle=False)) 

	return loaders, np.array(mesh_list)

def findNodeMappingForDataAugmentation(Nx, Ny):
	# For each node find the mapped node which has y-symmetric position
	# Compute first the regular lattice
	Lx = Nx*2
	Ly = Ny*np.sqrt(3)
	points = []
	for j in range(Ny):
		for i in range(Nx-(j+1)%2):
			p = [2*i + (j+1)%2, np.sqrt(3)*j]
			points.append(p)
	points = np.array(points)

	# Find mirrored points
	midY = np.mean(points[:,1])
	points[:,1] -= midY
	pointsMirrored = np.copy(points)
	pointsMirrored[:,1] *= -1

	# Compute mapping
	symMap = np.zeros(len(points)).astype(int)
	for i,p1 in enumerate(points):
		dist = np.linalg.norm(p1-pointsMirrored,axis=1)
		symMap[i] = np.argmin(dist)

	midY = np.c_[np.zeros(len(points)), np.full(len(points),midY)]

	return symMap, midY

def updateQuantitiesForNextStepPrediction(A0, A, X0, r, X_bounds, broken_edges, device):

	# We basically follow a similar process as in getData()
	broken_edges = np.array(broken_edges)
	last_broken_edge = broken_edges[-1]

	# Switch to numpy to do things as above
	A = A.detach().cpu().numpy()
	X = X0.clone().detach().cpu().numpy()
	r = r.detach().cpu().numpy()

	# Denormalize X
	X_min, X_range = X_bounds
	X = X*X_range + X_min

	# Remove corresponding edge features
	# (Before updating A to keep correct order)
	idx_nonzero_A = np.array(np.nonzero(A)).T
	intactEdgeIdxs = []
	for idx,e in enumerate(idx_nonzero_A):
		if list(e) == list(last_broken_edge): continue
		if list(e) == list(last_broken_edge)[::-1]: continue
		intactEdgeIdxs.append(idx)
	rC = r[intactEdgeIdxs]

	# Adjust the Z feature (reduced around crack)
	# Also compute mask based on first n broken edges
	# Only first neighbors are active in the mask 
	M = np.zeros_like(A)
	for e in idx_nonzero_A:
		if e[0] < e[1]: continue
		check0 = e[0] in broken_edges
		check1 = e[1] in broken_edges
		if check0 and not check1:
			X[e[1],-2] -= 1
			M[e[0],e[1]] = 1
			M[e[1],e[0]] = 1
		elif check1 and not check0:
			X[e[0],-2] -= 1
			M[e[0],e[1]] = 1
			M[e[1],e[0]] = 1

	# Update adjacency matrix
	A[broken_edges[:,0], broken_edges[:,1]] = 0 
	A[broken_edges[:,1], broken_edges[:,0]] = 0

	broken_nodes = set(np.unique(broken_edges.ravel()))
	broken_nodes_prev = broken_edges[:-1].ravel()
	# Use relative crack front position
	if broken_edges[-1,1] not in broken_nodes_prev:
		crackFrontPos = X[broken_edges[-1,1],:2]
	else:
		crackFrontPos = X[broken_edges[-1,0],:2]
	X[:,:2] -= crackFrontPos

	# Renormalize X
	X = (X-X_min)/X_range

	# Back to torch tensors
	return torch.tensor(A).to(device), \
		   torch.tensor(X, dtype=torch.float32).to(device), \
		   torch.tensor(rC).to(device), \
		   torch.tensor(M).to(device)
		   
def getNotchEdges(Nx, Ny, shape, notchWidth):

	# Data dir
	rootDir = '../data/'
	dataDirGDual = rootDir + str(shape) + '-' + str(Nx) + 'x' + str(Ny) + '-G-dual/'

	#=====================
	# Read dual mesh
	#=====================
	meshStr = dataDirGDual + 'mesh-0-dual.dat'
	nodes = []
	edges = []
	nodeOrEdge = 'node'
	with open(meshStr) as f:
		for line in f:
			line = line.rstrip('\n')
			if line=='connectivity':
				nodeOrEdge = 'edge'
			else:
				if nodeOrEdge == 'node':
					nodes.append([float(n) for n in line.split(',')])
				else:
					edges.append([int(n) for n in line.split(',')])

	nodes = np.array(nodes)
	edges = np.array(edges)

	# Convert to adjacency matrix
	A = np.zeros((len(nodes),len(nodes)))
	for e in edges:
		A[e[0]][e[1]] = 1
		A[e[1]][e[0]] = 1
	
	# Find notch nodes and edges
	notch_nodes = []
	node_idx = 0
	for j in range(Ny):
		for i in range(Nx-(j+1)%2):
			if j == (Ny-1)/2 and i < notchWidth: 
				notch_nodes.append(node_idx)
			node_idx += 1
		
	notch_edges = []
	for idxi,i in enumerate(notch_nodes):
		for j in notch_nodes[idxi:]:
			if A[i][j] > 0:
				notch_edges.append([i,j])

	return notch_edges

def readMesh(Nx, Ny, shape, meshIdx):
	# Data dir
	rootDir = '../data/'
	dataDirGDual = rootDir + str(shape) + '-' + str(Nx) + 'x' + str(Ny) + '-G-dual/'
	meshStr = dataDirGDual + 'mesh-' + str(meshIdx) + '-dual.dat'
	nodes = []
	edges = []
	nodeOrEdge = 'node'
	with open(meshStr) as f:
		for line in f:
			line = line.rstrip('\n')
			if line=='connectivity':
				nodeOrEdge = 'edge'
			else:
				if nodeOrEdge == 'node':
					nodes.append([float(n) for n in line.split(',')])
				else:
					edges.append([int(n) for n in line.split(',')])

	nodes = np.array(nodes)
	edges = np.array(edges)
	
	return nodes, edges
