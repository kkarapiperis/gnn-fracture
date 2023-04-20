import sys
import time
import random
import numpy as np
import scipy.spatial as sp
import matplotlib.pyplot as plt
from voronoiUtilities import *

class Voronoi:

    def __init__(self, Nx, Ny, shape, notchWidth, lamda, seed=1):
        """
        Nx,Ny: number of points along each dim
        notchWidth: number of missing cells forming the notch
        shape: shape of the lattice
        lamda: topological paarameters
        """
        random.seed(seed)
        np.random.seed(seed)

        self.Nx = Nx
        self.Ny = Ny
        self.seed = seed
        self.shape = shape
        self.lamdaZ = lamda[0]
        self.lamdaGeom = lamda[1]
        self.notchWidth = notchWidth

        if shape == 'hex':
            self.generateHexLattice()
            self.Zhcp = 6
        else:
            print('Unknown shape!')
            sys.exit()
        
    def generateHexLattice(self):
        ''' 
        Create 2D Hexagonal lattice
        '''
        self.Lx = self.Nx*2
        self.Ly = self.Ny*np.sqrt(3)
        self.points = []
        for j in range(self.Ny):
            for i in range(self.Nx-(j+1)%2):
                p = [2*i + (j+1)%2, np.sqrt(3)*j]
                self.points.append(p)
        self.spacing = 2
        self.points = np.array(self.points)

    def sampleNuclei(self):
        '''
        Run statistical model to find nuclei positions
        '''
        t0 = time.time()

        model = Ensemble(self.Nx, self.Ny, self.notchWidth, self.points, 
            self.Zhcp, self.spacing, self.lamdaZ, self.lamdaGeom, self.seed)

        # Enough steps for equilibration 
        nSteps = int(0.5*pow(len(self.points),2))

        self.E = np.zeros(nSteps)
        for step in range(nSteps):
            self.E[step] = model.iterate()

        # Get equilibrated configuration
        self.points = model.points 
        
        # Get individual Hamiltonians
        self.voronoiDegrees = model.getDegrees()
        self.HZ, self.HGeom = model.getHamiltonians(model.points)
        self.HZ /= len(self.points)
        self.HGeom /= len(self.points)
        self.computeVoronoiTesselation()

        t1 = time.time()
        print('Hz = ', self.HZ)
        print('HGeom = ', self.HGeom)
        print('Performed MCMC computation (', t1-t0, ' s - ', nSteps, ' steps)')

    def loadConfig(self, idx):
        '''
        Load nuclei positions from the dual mesh file
        '''
        dataDir = '../data/' + self.shape + '-' + str(self.Nx)+'x'+str(self.Ny) + '-G-dual'
        filename = dataDir + '/mesh-' + str(idx) + '-dual.dat'
        points,_ = loadLattice(filename)
        self.points = np.copy(points)

        # Compute the hamiltonians
        model = Ensemble(self.Nx, self.Ny, self.notchWidth, self.points, 
            self.Zhcp, self.spacing, self.lamdaZ, self.lamdaGeom, self.seed)
        model.setPoints(self.points)
        self.voronoiDegrees = model.getDegrees()
        self.HZ, self.HGeom = model.getHamiltonians(self.points)
        self.HZ /= len(self.points)
        self.HGeom /= len(self.points)
        self.computeVoronoiTesselation()

    def computeVoronoiTesselation(self):
        '''
        Perform the tesselation on the seed points
        '''
        # Add ghost points for periodicity
        points = addGhostPoints(self.points, self.Nx, self.Ny)

        # Do the Voronoi computation
        self.voronoi = sp.Voronoi(points,qhull_options="Qbb Qx")

        # Correct the tesselation by removing any near-duplicate vertices
        self.voronoiVerts, self.voronoiRidgeVerts, self.voronoiRidgePoints, \
            self.voronoiRidgeLengths, self.voronoiRegions = correctTesselation(
                self.points, self.voronoi)

        # Check for any duplicate nodes or edges
        assert(not checkForDuplicateNodes(self.voronoiVerts))
        assert(not checkForDuplicateEdges(self.voronoiRidgeVerts))

        # Compute lower scale voronoi features (anisotropy vector,shape vector,volume)
        self.voronoiBarycenters, self.voronoiAnisotropyVectors, self.voronoiVolumes, \
            self.voronoiMajorAxes, self.voronoiCircularity, self.voronoiMinEdgeLength = \
            computeVoronoiFeatures(self.points, self.voronoiVerts, self.voronoiRegions)

    def saveGraph(self, meshIdx):
        '''
        Save nodes and edges for use in downstream FEM 
        '''
        dataDir = '../data/' + self.shape + '-' + str(self.Nx)+'x'+str(self.Ny)+'-G'
        f = open(dataDir + '/mesh-' + str(meshIdx) +'.dat','w')
        for node in self.voronoiVerts:
            f.write('%.8f,%.8f\n' % (node[0], node[1])) 
        f.write('connectivity\n')
        for e in self.voronoiRidgeVerts:
            f.write('%d,%d\n' % (e[0], e[1])) 
        f.close()

    def appendToInputList(self, meshIdx):
        """
        Append the specific input that generated the mesh to the list of all inputs
        """
        dataDir = '../data/' + self.shape + '-' + str(self.Nx)+'x'+str(self.Ny) + '-G'
        filename = dataDir + '/mesh-list.dat'
        if meshIdx == 0:
            f = open(filename, "w")
            f.write('lamdaZ, ') 
            f.write('lamdaGeom, ') 
            f.write('HZ, ') 
            f.write('HGeom, ') 
            f.write('seed\n') 
            f.write(str(round(self.lamdaZ,3)) + ', ')
            f.write(str(round(self.lamdaGeom,3)) + ', ')
            f.write(str(round(self.HZ,3)) + ', ')
            f.write(str(round(self.HGeom,3)) + ', ')
            f.write(str(self.seed) + '\n')
        else:
            f = open(filename, "a")
            f.write(str(round(self.lamdaZ,3)) + ', ')
            f.write(str(round(self.lamdaGeom,3)) + ', ')
            f.write(str(round(self.HZ,3)) + ', ')
            f.write(str(round(self.HGeom,3)) + ', ')
            f.write(str(self.seed) + '\n')
        f.close() 

    def saveDualGraph(self, meshIdx):
        '''
        Save the graph of the voronoi nodes along with features
        of interest for use in downstream graph learning
        '''
        dataDir = '../data/' + self.shape + '-' + str(self.Nx)+'x'+str(self.Ny)+'-G-dual'
        f = open(dataDir + '/mesh-' + str(meshIdx) + '-dual.dat','w')
        for node in self.points:
            f.write('%.4f,%.4f\n' % (node[0], node[1])) 
        f.write('connectivity\n')
        for e in self.voronoiRidgePoints:
            f.write('%d,%d\n' % (e[0], e[1])) 
        f.close()
        # Save also nodal quantities
        f = open(dataDir + '/mesh-' + str(meshIdx) + '-X.dat','w')
        for idx,node in enumerate(self.points):
            f.write('%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n' % 
                (self.voronoiVolumes[idx], 
                 self.voronoiAnisotropyVectors[idx][0], 
                 self.voronoiAnisotropyVectors[idx][1], 
                 self.voronoiMajorAxes[idx][0],
                 self.voronoiMajorAxes[idx][1],
                 self.voronoiCircularity[idx],
                 self.voronoiMinEdgeLength[idx],
                 self.voronoiDegrees[idx])) 
        f.close()
        # Save also edge attributes
        # Find adjacency matrix (auxilliary computation)
        A = np.zeros((len(self.points),len(self.points)))
        for e in self.voronoiRidgePoints:
            A[e[0]][e[1]] = 1
            A[e[1]][e[0]] = 1
        # Compute relative node positions, primal edge lengths
        relPos = []
        edgeLengths = []
        for idxI in range(len(self.points)):
            for idxJ in range(len(self.points)):
                if A[idxI][idxJ] == 1:
                    # Find ridge index 
                    relPos.append(self.points[idxJ]-self.points[idxI])
                    # Compute length of corresponding primal edge 
                    try:
                        idxIJ = self.voronoiRidgePoints.index([idxI,idxJ])
                    except ValueError:
                        idxIJ = self.voronoiRidgePoints.index([idxJ,idxI])
                    edgeLengths.append(self.voronoiRidgeLengths[idxIJ])
        
        f = open(dataDir + '/mesh-' + str(meshIdx) + '-E.dat','w')
        for idx in range(len(relPos)):
            f.write('%.4f,%.4f,%.4f\n' % 
                (relPos[idx][0], relPos[idx][1], edgeLengths[idx])) 
        f.close()

    def plotLattice(self):
        fig = plt.figure(figsize=(0.5*self.Nx,0.5*self.Ny))
        ax = fig.add_subplot()
        ax.scatter(self.voronoiVerts[:, 0], self.voronoiVerts[:, 1], s=5, c="r")
        ax.scatter(self.points[:,0],self.points[:,1],s=10, c="k")
        for e in self.voronoiRidgeVerts:
            ax.plot(self.voronoiVerts[e,0],self.voronoiVerts[e,1], c='k')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(-3,self.Lx+2)
        ax.set_ylim(-3,self.Ly+2)
        plt.show(block=True)

    def plotDualLattice(self):
        fig = plt.figure(figsize=(0.5*self.Nx,0.5*self.Ny))
        ax = fig.add_subplot()
        for e in self.voronoiRidgePoints:
            ax.plot(self.points[e,0],self.points[e,1], c='k')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(-3,self.Lx+2)
        ax.set_ylim(-3,self.Ly+2)
        plt.show(block=True)