import sys
import time
import random
import itertools
import numpy as np
import scipy.spatial as sp
import matplotlib.pyplot as plt
sys.path.append('../generation')
from voronoiUtilities import *

class Voronoi:

    def __init__(self, Nx, Ny, shape, notchWidth, lamda, hexSpacing, hexZ, seed=1):
        """
        Nx,Ny: number of points along each dim
        points: initial set of points
        """
        self.Nx = Nx
        self.Ny = Ny
        self.shape = shape
        self.seed = seed
        self.hexZ = hexZ
        self.lamdaZ = lamda[0]
        self.lamdaGeom = lamda[1]
        self.hexSpacing = hexSpacing
        self.pointsRegular = self.generateHexLattice()
        self.points = np.copy(self.pointsRegular)
        self.bdIdx = getBoundaryNucleiIndices(Nx, Ny, notchWidth)
        self.centroid = np.mean(self.pointsRegular,axis=0)
        self.N = len(self.pointsRegular)
        self.intIdx = list(set(range(self.N)).difference(self.bdIdx))
        self.bdPts = self.pointsRegular[self.bdIdx]
        self.initializeHot()

    def generateHexLattice(self):
        ''' 
        Create 2D Hexagonal lattice
        '''
        self.Lx = self.Nx*2
        self.Ly = self.Ny*np.sqrt(3)
        pointsRegular = []
        for j in range(self.Ny):
            for i in range(self.Nx-(j+1)%2):
                p = [2*i + (j+1)%2, np.sqrt(3)*j]
                pointsRegular.append(p)
        return np.array(pointsRegular)

    def singleBdConformingMove(self, idx):
        """
        Return random displacement that respects overall boundary
        """        
        theta = np.random.uniform(-np.pi,np.pi)
        dispMag = np.random.uniform(0, 0.1*self.hexSpacing)
        disp = [dispMag*np.cos(theta), dispMag*np.sin(theta)]
        distBd = np.linalg.norm(self.bdPts-self.points[idx],axis=1)
        distBdCentroid = np.linalg.norm(self.bdPts[np.argmin(distBd)]-self.centroid)
        distPtCentroid = np.linalg.norm(self.points[idx]+disp-self.centroid)
        checkBd = distPtCentroid < distBdCentroid
        return disp if checkBd else [0,0]

    def initializeHot(self):
        """
        Initialize in a random state
        """
        for idx in self.intIdx:            
            self.points[idx] += self.singleBdConformingMove(idx)       

    def iterate(self):
        """
        Single MC iteration
        """
        # Store current configuration in case the current iterate
        # is later rejected and we need to revert back
        self.points_prev = np.copy(self.points)

        # Find a boundary conforming move
        idx = np.random.choice(self.intIdx)
        disp = self.singleBdConformingMove(idx)
        self.points[idx] += disp

    def equilibrate(self):
        """
        Call iterate() above until equilibration in energy sense
        """
        # Steps for equilibration 
        nSteps = int(0.8*pow(len(self.points),2))

        energy = self.getEnergy()
        for step in range(nSteps):
            self.iterate()
            newEnergy = self.getEnergy()
            dE = newEnergy - energy
            p = np.random.uniform()
            r = np.exp(-dE/self.N)
            if p < r:                
                energy = np.copy(newEnergy)
            else:
                self.revert()

    def getEnergy(self):
        """
        Return energy  
        """
        start = time.time()
        voronoiRidgePoints,numGhostPts = self.computeVoronoiTesselationRidgePoints()

        # Initialize 
        Z = np.zeros(self.N+numGhostPts)
        E = 0
        for i,j in voronoiRidgePoints:
            Z[i] += 1; Z[j] += 1
            if i >= self.N: continue
            if j >= self.N: continue
            dIJ = np.linalg.norm(self.points[i]-self.points[j])
            E += self.lamdaGeom*abs(dIJ-self.hexSpacing)

        for i in range(self.N):
            E += self.lamdaZ*pow(Z[i]-self.hexZ,2)

        return E

    def computeDegreesAndDisorder(self):
        """
        Compute nodal degrees accounting for periodicity in the boundaries
        """
        voronoiRidgePoints,numGhostPts = self.computeVoronoiTesselationRidgePoints()

        # Initialize 
        Z = np.zeros(self.N+numGhostPts)
        self.HZ = 0
        self.HGeom = 0
        for i,j in voronoiRidgePoints:
            Z[i] += 1; Z[j] += 1
            if i >= self.N: continue
            if j >= self.N: continue
            dIJ = np.linalg.norm(self.points[i]-self.points[j])
            self.HGeom += abs(dIJ-self.hexSpacing)

        for i in range(self.N):
            self.HZ += pow(Z[i]-self.hexZ,2)

        self.voronoiDegrees = []
        for i in range(self.N):
            self.voronoiDegrees.append(Z[i])

        self.HZ /= self.Nx*self.Ny
        self.HGeom /= self.Nx*self.Ny

    def revert(self):
        """
        Revert updating of nuclei
        """
        self.points = np.copy(self.points_prev)

    def computeVoronoiTesselationRidgePoints(self):
        """
        Return only the tesselation data needed for energy
        """
        # Add ghost points for periodicity
        points = addGhostPoints(self.points, self.Nx, self.Ny)
        numGhostPts = len(points) - self.N
        voronoi = sp.Voronoi(points,qhull_options="Qbb Qx")
        voronoiRidgePoints = correctTesselationRidgePoints(self.points, voronoi)
        return voronoiRidgePoints, numGhostPts

    def computeVoronoiTesselation(self):
        '''
        Same as above but returns also voronoi nodal features
        '''
        # Add ghost points for periodicity
        points = addGhostPoints(self.points, self.Nx, self.Ny)
        voronoi = sp.Voronoi(points,qhull_options="Qbb Qx")

        # Correct the tesselation by removing any near-duplicate vertices
        self.voronoiVerts, self.voronoiRidgeVerts, self.voronoiRidgePoints, \
            self.voronoiRidgeLengths, self.voronoiRegions = correctTesselation(
                self.points, voronoi)

        # Check for any duplicate nodes or edges
        assert(not checkForDuplicateNodes(self.voronoiVerts))
        assert(not checkForDuplicateEdges(self.voronoiRidgeVerts))

        # Compute lower scale voronoi features (anisotropy vector,shape vector,volume)
        self.voronoiBarycenters, self.voronoiAnisotropyVectors, self.voronoiVolumes, \
            self.voronoiMajorAxes, self.voronoiCircularity, self.voronoiMinEdgeLength = \
            computeVoronoiFeatures(self.points, self.voronoiVerts, self.voronoiRegions)

        self.voronoiBarycenters = np.array(self.voronoiBarycenters)
        self.voronoiAnisotropyVectors = np.array(self.voronoiAnisotropyVectors)
        self.voronoiMajorAxes = np.array(self.voronoiMajorAxes)

        self.computeDegreesAndDisorder()

    def loadConfig(self, idx):
        '''
        Load nuclei positions from a dual mesh file
        '''
        dataDir = '../data/' + self.shape + '-' + str(self.Nx)+'x'+str(self.Ny) + '-G-dual'
        filename = dataDir + '/mesh-' + str(idx) + '-dual.dat'
        points,_ = loadLattice(filename)
        self.points = np.copy(points)

    def saveGraph(self, saveName):
        '''
        Save nodes and edges for use in downstream FEM 
        '''
        f = open(saveName + '-opt_mesh.dat','w')
        for node in self.voronoiVerts:
            f.write('%.8f,%.8f\n' % (node[0], node[1])) 
        f.write('connectivity\n')
        for e in self.voronoiRidgeVerts:
            f.write('%d,%d\n' % (e[0], e[1])) 
        f.close()

    def saveDualGraph(self, saveName):
        '''
        Save the voronoi dual graph graph 
        '''
        f = open(saveName + '-opt_mesh_dual.dat','w')
        for node in self.points:
            f.write('%.4f,%.4f\n' % (node[0], node[1])) 
        f.write('connectivity\n')
        for e in self.voronoiRidgePoints:
            f.write('%d,%d\n' % (e[0], e[1])) 
        f.close()

    def getDualGraphAndFeatures(self, notchWidth):

        # Compute adjacency matrix 
        A = np.zeros((self.N,self.N))
        for e in self.voronoiRidgePoints:
            A[e[0]][e[1]] = 1
            A[e[1]][e[0]] = 1

        # Compute nodal and edge features
        nodes = self.points
        X = np.c_[self.voronoiVolumes, 
                  self.voronoiAnisotropyVectors[:,0], 
                  self.voronoiAnisotropyVectors[:,1], 
                  self.voronoiMajorAxes[:,0],
                  self.voronoiMajorAxes[:,1],
                  self.voronoiCircularity,
                  self.voronoiMinEdgeLength,
                  self.voronoiDegrees]

        # Adjust those features exactly as used in training
        # Volume, magnitude of anisotropy, major axis orientation, 
        # circularity, min edge length, Z, order parameter
        XMagAnisotr = np.linalg.norm(X[:,1:3],axis=1)
        # Make sure major axis is pointing towards +x when computing theta
        XMajAxis = np.einsum('ij,i->ij',X[:,3:5], np.sign(X[:,3]+1e-6))
        XMajAxisTheta = np.arctan2(XMajAxis[:,0],XMajAxis[:,1])
        # Order parameter as geometric clustering feature (Janssens et al, 2006)
        op = np.zeros(self.N)
        for i in range(self.N):
            for j in range(self.N):
                if A[i][j] == 1:
                    op[i] += np.linalg.norm(nodes[i]-nodes[j]) 
        X = np.c_[1/X[:,0], XMagAnisotr, XMajAxisTheta, X[:,-3], 1/X[:,-2], X[:,-1],op]
        X = np.hstack((nodes,X))

        # Save also edge attributes (relative node positions, edge lengths)
        relPos = []
        edgeLengths = []
        for idxI in range(self.N):
            for idxJ in range(self.N):
                if A[idxI][idxJ] == 1:
                    # Find ridge index 
                    relPos.append(self.points[idxJ]-self.points[idxI])
                    # Compute length of corresponding primal edge 
                    try:
                        idxIJ = self.voronoiRidgePoints.index([idxI,idxJ])
                    except ValueError:
                        idxIJ = self.voronoiRidgePoints.index([idxJ,idxI])
                    edgeLengths.append(self.voronoiRidgeLengths[idxIJ])
        
        r = np.c_[relPos, edgeLengths]
        # Switch to magnitude and orientation of the relative node positions 
        relPos = np.einsum('ij,i->ij',r[:,:2], np.sign(r[:,0]+1e-6))
        r[:,0] = np.arctan2(relPos[:,1],relPos[:,0])
        r[:,1] = np.linalg.norm(relPos,axis=1)

        # Adjust features due to the notch
        all_nodes = set(range(len(nodes)))

        # Find notch nodes and define a helpful function for later
        notch_nodes = []
        node_idx = 0
        for j in range(self.Ny):
            for i in range(self.Nx-(j+1)%2):
                if j == (self.Ny-1)/2 and i < notchWidth: 
                    notch_nodes.append(node_idx)
                node_idx += 1

        # Find their neighbors
        def findNotchNeighbors(notchNodes):
            neighbors = []
            for i in notchNodes:
                for j in range(len(nodes)):
                    if A[i][j] == 0: continue
                    if j in notchNodes: continue
                    if j in neighbors: continue
                    neighbors.append(j) 
            return neighbors
            
        # To start off the sequence, add the existing notch edges as broken edges
        notch_edges = []
        for idxi,i in enumerate(notch_nodes):
            for j in notch_nodes[idxi:]:
                if A[i][j] > 0:
                    notch_edges.append([i,j])

        broken_edges = np.copy(notch_edges[::-1])

        # Find the updated adjacency matrix accounting for broken edges
        n = len(notch_edges)
        AC = np.copy(A)
        AC[broken_edges[:n,0], broken_edges[:n,1]] = 0 
        AC[broken_edges[:n,1], broken_edges[:n,0]] = 0
        # Remove corresponding edge features
        rIdx = -1
        intactEdgeIdxs = []
        # Remove corresponding edge features
        # rIdx corresponds to the order in the uncracked indices
        for idxI in range(len(nodes)):
            for idxJ in range(len(nodes)):
                if A[idxI][idxJ] > 0:
                    rIdx += 1
                    if AC[idxI][idxJ] > 0:
                        intactEdgeIdxs.append(rIdx)
        rC = r[intactEdgeIdxs]
        # Adjust the Z feature (reduced around crack)
        XC = np.copy(X)
        notch_nodes = np.unique(broken_edges[:n].ravel())
        notch_node_neighbors = findNotchNeighbors(notch_nodes)
        for i in notch_node_neighbors:
            for j in notch_nodes:
                if A[i][j] == 0: continue
                XC[i,-2] -= 1
        # Create mask based on first n broken edges
        # Obviously start with the adjacency matrix
        # If edge does not exist, it cannot break (mask=0)
        M = np.copy(AC)
        broken_nodes = set(np.unique(broken_edges[:n].ravel()))
        intact_nodes = list(all_nodes.difference(broken_nodes))
        idx = tuple(np.array(list(itertools.combinations(intact_nodes,2))).T)
        M[idx[::-1]] = 0
        M[idx] = 0

        # Use relative crack front
        broken_nodes_prev = broken_edges[:n-1].ravel()
        if broken_edges[n-1,1] not in broken_nodes_prev:
            crackFrontPos = nodes[broken_edges[n-1,1],:2]
        else:
            crackFrontPos = nodes[broken_edges[n-1,0],:2]
        XC[:,:2] -= crackFrontPos

        return AC, XC, rC, M
