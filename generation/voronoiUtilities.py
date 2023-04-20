import math
import numpy as np
import networkx as nx
import scipy.spatial as sp

def addNode(p,nodes,tol=1e-1):
    """
    Auxilliary function for adding node to a preexisting list
    """
    for idx,n in enumerate(nodes):
        if abs(p[0]-n[0]) > tol: continue
        if abs(p[1]-n[1]) > tol: continue
        return nodes, idx
    nodes.append(p)
    return nodes, len(nodes)-1

def checkForDuplicateNodes(nodes,tol=1e-3):
    """
    Returns true if we find duplicate entries
    """
    for i,node in enumerate(nodes):
        for node2 in nodes[i+1:]:
            dist = np.linalg.norm(node-node2)
            if dist < tol:
                return True
    return False

def checkForDuplicateEdges(edges):
    """
    Returns true if we find duplicate entries
    """
    for i,edge in enumerate(edges):
        for edge2 in edges[i+1:]:
            if edge[0] == edge2[0] and edge[1] == edge2[1]: 
                return True
            if edge[0] == edge2[1] and edge[1] == edge2[0]: 
                return True
    return False

def findCentroidAndAreaOfPolygon(verts):
    """
    Returns area, centroid, minimum edge length of planar polygon in 3D
    """
    n = len(verts)
    centroid = np.mean(verts,axis=0)

    # Define sorting function based 
    def clockwiseAngleAndDistance(point):
        refvec = [0, 1]
        vector = [point[0]-centroid[0], point[1]-centroid[1]]
        lenvector = math.hypot(vector[0], vector[1])
        if lenvector == 0:
            return -math.pi, 0
        normalized = [vector[0]/lenvector, vector[1]/lenvector]
        dotprod = normalized[0]*refvec[0] + normalized[1]*refvec[1]    
        diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]   
        angle = math.atan2(diffprod, dotprod)
        if angle < 0:
            return 2*math.pi+angle, lenvector
        return angle, lenvector

    verts = sorted(verts,key=clockwiseAngleAndDistance)
    barycenter = np.zeros(2)
    minEdgeLength = 1e8
    polyArea = 0
    for i in range(n):
        triangle = [verts[i],verts[(i+1)%n],centroid]
        triangleCentroid = np.mean(triangle,axis=0)
        cross = np.cross(verts[i]-centroid, verts[(i+1)%n]-centroid)
        triangleArea = 0.5*np.linalg.norm(cross)
        barycenter += triangleArea*triangleCentroid
        polyArea += triangleArea
        edgeLength = np.linalg.norm(verts[i]-verts[(i+1)%n])
        minEdgeLength = min(minEdgeLength,edgeLength)
    return barycenter/polyArea, polyArea, minEdgeLength

def addGhostPoints(points, Nx, Ny):
    """
    Add exterior layer of ghost points for tesselation
    """
    points = list(points)
    for j in [-1,Ny]:
        for i in range(-1,Nx-(j+1)%2+1):
            p = [2*i + (j+1)%2, np.sqrt(3)*j]
            points.append(p)

    for j in range(Ny):
        for i in [-1,Nx-(j+1)%2]:
            p = [2*i + (j+1)%2, np.sqrt(3)*j]
            points.append(p)

    return points

def correctTesselation(points, voronoi):
    """
    Remove any duplicate vertices and any boundary artifacts
    """
    # Initialize
    voronoiVerts = []
    voronoiRidgeVerts = []
    voronoiRidgePoints = []
    voronoiRidgeLengths = []
    voronoiRegions = [[] for p in points]
    nNuclei = len(points)

    for ridgeIdx,ridge in enumerate(voronoi.ridge_vertices):
        # Disregard if not around the kept nuclei
        nucleiIdx0 = voronoi.ridge_points[ridgeIdx][0]
        nucleiIdx1 = voronoi.ridge_points[ridgeIdx][1]
        check0 = nucleiIdx0 < nNuclei
        check1 = nucleiIdx1 < nNuclei
        if not check0 and not check1: continue
        # Add vertex
        vert0 = voronoi.vertices[ridge[0]]
        vert1 = voronoi.vertices[ridge[1]]
        voronoiVerts,vertIdx0 = addNode(vert0,voronoiVerts)
        voronoiVerts,vertIdx1 = addNode(vert1,voronoiVerts)
        vertIdx0,vertIdx1 = min(vertIdx0,vertIdx1),\
                            max(vertIdx0,vertIdx1)
        # Add to corresponding region
        if check0:
            if vertIdx0 not in voronoiRegions[nucleiIdx0]:
                voronoiRegions[nucleiIdx0].append(vertIdx0)
            if vertIdx1 not in voronoiRegions[nucleiIdx0]:
                voronoiRegions[nucleiIdx0].append(vertIdx1)
        if check1:
            if vertIdx0 not in voronoiRegions[nucleiIdx1]:
                voronoiRegions[nucleiIdx1].append(vertIdx0)
            if vertIdx1 not in voronoiRegions[nucleiIdx1]:
                voronoiRegions[nucleiIdx1].append(vertIdx1)
        # Add ridge
        if vertIdx0 == vertIdx1: continue
        if [vertIdx0,vertIdx1] in voronoiRidgeVerts: continue
        voronoiRidgeVerts.append([vertIdx0,vertIdx1])
        if not check0 or not check1: continue 
        voronoiRidgePoints.append([nucleiIdx0,nucleiIdx1])
        voronoiRidgeLengths.append(np.linalg.norm(vert1-vert0))

    return np.array(voronoiVerts), np.array(voronoiRidgeVerts), \
        voronoiRidgePoints, voronoiRidgeLengths, voronoiRegions

def correctTesselationRidgePoints(points, voronoi):
    """
    Same as above but only returns ridge points
    """
    nNuclei = len(points)
    mask = (voronoi.ridge_points<nNuclei).any(1) 
    return voronoi.ridge_points[mask]

def loadLattice(fileName):
    """
    Load mesh from a file
    """
    # Initialize
    nodes = [] 
    edges = []
    nodeOrEdge = 'node'

    with open(fileName) as f:
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

    return nodes,edges

def readEdgeList(filename):
   edge_list = []
   with open(filename) as f:
      for line in f.readlines():
         edges= []
         for edge_str in line.strip().split(", "):
            edge_str = edge_str.split(",")
            edge = [int(edge_str[0][1:]),
                   int(edge_str[1][:-1])]
            edges.append(edge)
         edge_list.append(edges)
         
   return np.array(edge_list, dtype=object)

def readMeshFile(filename):
    nodes = []
    edges = []
    nodeOrEdge = 'node'
    with open(filename) as f:
        for line in f:
            line = line.rstrip('\n')
            if line=='connectivity':
                nodeOrEdge = 'edge'
            else:
                if nodeOrEdge == 'node':
                    nodes.append([float(n) for n in line.split(',')])
                else:
                    edges.append([int(n) for n in line.split(',')])
    return np.array(nodes), edges

def getBoundaryNucleiIndices(Nx,Ny,notchWidth):
    """
    Return indices of the nuclei at the boundaries
    """
    bdNucleiIdx = []
    idx = 0
    for j in range(Ny):
        for i in range(Nx-(j+1)%2):
            if j%(Ny-1) == 0: 
                bdNucleiIdx.append(idx)
            elif i%(Nx-(j+1)%2) == 0:
                bdNucleiIdx.append(idx)
            elif i%(Nx-1-(j+1)%2) == 0:
                bdNucleiIdx.append(idx)
            elif j == (Ny-1)/2 and i < notchWidth: 
                bdNucleiIdx.append(idx)
            idx += 1
    return bdNucleiIdx

def getRightBoundaryNucleiIndices(Nx,Ny):
    """
    Return indices of the nuclei at the boundaries
    """
    bdNucleiIdx = []
    idx = 0
    for j in range(Ny):
        for i in range(Nx-(j+1)%2):
            if i == Nx-1-(j+1)%2:
                bdNucleiIdx.append(idx)
            idx += 1
    return bdNucleiIdx
    
def computeVoronoiFeatures(points, voronoiVerts, voronoiRegions):
    """
    Compute Voronoi cell features: Anisotropy vector (connecting 
    barycenter and nucleus), shape vector (principal axis), volume
    """
    # Initialize
    voronoiVolumes = []
    voronoiBarycenters = []
    voronoiMajorAxes = []
    voronoiCircularity = []
    voronoiMinEdgeLength = []
    for ptIdx,p in enumerate(points):
        vertIdxs = voronoiRegions[ptIdx]
        verts = [voronoiVerts[i] for i in vertIdxs]
        centroid, cellVolume, minEdgeLength = \
            findCentroidAndAreaOfPolygon(verts)
        shapeTensor = np.zeros((2,2))
        for v in verts:
            vrel = v-centroid
            shapeTensor += np.outer(vrel,vrel)/len(verts)
        eigVals, eigVecs = np.linalg.eigh(shapeTensor)
        if np.dot(eigVecs[1],[0,1]) > 1e-8:
            eigVecs[1] = eigVecs[1]*np.dot(eigVecs[1],[0,1])
            eigVecs[1] /= np.linalg.norm(eigVecs[1])
        voronoiBarycenters.append(centroid) 
        voronoiVolumes.append(cellVolume)
        voronoiMajorAxes.append(eigVecs[1])
        voronoiCircularity.append(eigVals[1]/eigVals[0])
        voronoiMinEdgeLength.append(minEdgeLength)

    voronoiBarycenters = np.array(voronoiBarycenters)
    voronoiVolumes = np.array(voronoiVolumes)
    voronoiAnisotropyVectors = voronoiBarycenters - points

    return voronoiBarycenters, voronoiAnisotropyVectors, voronoiVolumes, \
        voronoiMajorAxes, voronoiCircularity, voronoiMinEdgeLength

class Ensemble:

    def __init__(self, Nx, Ny, notchWidth, points, Zhcp, hexDist, lamdaZ, lamdaGeom, seed):
        """
        Arguments:
        points: voronoi nuclei
        lamdaZ: coupling coefficient for topological energy term (Z)
        lamdaGeom: coupling coefficient for geometrical energy term
        """
        np.random.seed(seed)
        self.Nx = Nx
        self.Ny = Ny
        self.notchWidth = notchWidth
        self.points = points
        self.N = len(points)
        self.Zhcp = Zhcp
        self.hexDist = hexDist
        self.lamdaZ = lamdaZ
        self.lamdaGeom = lamdaGeom 
        self.initializeHot()
        
    def randomBdConformingMove(self, idx):
        """
        Return random displacement within ball that respects overall boundary
        """        
        theta = np.random.uniform(-np.pi,np.pi)
        dispMag = np.random.uniform(0, 0.15*self.hexDist)
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
        self.pointsCold = self.points.copy()
        self.centroid = np.mean(self.pointsCold,axis=0)
        self.bdIdx = getBoundaryNucleiIndices(self.Nx, self.Ny, self.notchWidth)
        self.intIdx = list(set(range(self.N)).difference(self.bdIdx))
        self.bdPts = self.points[self.bdIdx]
        for idx in self.intIdx:            
            self.points[idx] += self.randomBdConformingMove(idx)       
        self.energy = self.getEnergy(self.points)

    def iterate(self):
        """
        Single MC iteration 
        """
        idx = np.random.choice(self.intIdx)
        disp = self.randomBdConformingMove(idx)
        if np.linalg.norm(disp) < 1e-6: return self.energy
        newPoints = self.points.copy()
        newPoints[idx] += disp       
        newEnergy = self.getEnergy(newPoints)
        dE = newEnergy - self.energy
        p = np.random.uniform()
        r = np.exp(-dE/self.N)
        if p < r:
            self.energy = newEnergy
            self.points = newPoints

        return self.energy

    def setPoints(self, points):
        """
        Set positions
        """
        self.points = points

    def getEnergy(self, points):
        """
        Return energy  
        """
        voronoiRidgePoints = self.computeTesselation(points)

        # Create auxillliary graph object
        G = nx.Graph()
        G.add_nodes_from(range(self.N))
        G.add_edges_from(voronoiRidgePoints)

        # Initialize 
        E = 0
        for i,j in voronoiRidgePoints:
            if i >= self.N: continue
            if j >= self.N: continue
            dIJ = np.linalg.norm(points[i]-points[j])
            E += self.lamdaGeom*abs(dIJ-self.hexDist)

        for i in range(self.N):
            ZI = G.degree[i]
            E += self.lamdaZ*pow(ZI-self.Zhcp,2)

        return E

    def getHamiltonians(self, points):
        """
        Return current configuration's H terms  
        """
        voronoiRidgePoints = self.computeTesselation(points)

        # Create auxillliary graph object
        G = nx.Graph()
        G.add_nodes_from(range(self.N))
        G.add_edges_from(voronoiRidgePoints)

        # Initialize 
        HZ = 0
        HGeom = 0

        for i,j in voronoiRidgePoints:
            if i >= self.N: continue
            if j >= self.N: continue
            dIJ = np.linalg.norm(points[i]-points[j])
            HGeom += abs(dIJ-self.hexDist)

        for i in range(self.N):
            ZI = G.degree[i]
            HZ += pow(ZI-self.Zhcp,2)

        return HZ, HGeom

    def getDegrees(self):
        """
        Return vector of node degree
        """
        voronoiRidgePoints = self.computeTesselation(self.points)

        # Create auxillliary graph object
        G = nx.Graph()
        G.add_nodes_from(range(self.N))
        G.add_edges_from(voronoiRidgePoints)

        Z = []
        for i in range(self.N):
            Z.append(G.degree[i])

        return np.array(Z)

    def computeTesselation(self, points):
        """
        Return tesselation data needed for energy
        """
        points = addGhostPoints(points, self.Nx, self.Ny)
        voronoi = sp.Voronoi(points,qhull_options="Qbb Qx")
        voronoiRidgePoints = self.restrictTesselation(self.points, voronoi)
        return voronoiRidgePoints

    def restrictTesselation(self, points, voronoi):
        """
        Restrict to interior nuclei
        """
        # Initialize
        voronoiRidgePoints = []
        nNuclei = len(points)

        for ridgeIdx,ridge in enumerate(voronoi.ridge_vertices):
            # Disregard if not around the kept nuclei
            nucleiIdx0 = voronoi.ridge_points[ridgeIdx][0]
            nucleiIdx1 = voronoi.ridge_points[ridgeIdx][1]
            check0 = nucleiIdx0 < nNuclei
            check1 = nucleiIdx1 < nNuclei
            if not check0 and not check1: continue
            nucleiIdx0,nucleiIdx1 = min(nucleiIdx0,nucleiIdx1), \
                                    max(nucleiIdx0,nucleiIdx1)
            if [nucleiIdx0,nucleiIdx1] in voronoiRidgePoints: continue
            voronoiRidgePoints.append([nucleiIdx0,nucleiIdx1])
        return voronoiRidgePoints
