import math
import numpy
import random
import sys

import pprint


class RegulatoryNetwork:
    '''
    Implimentation of analog Hopfield neural network model for genetic
    regulatory networks. This network represents the morphogens acting on
    and produced by a single cell. Multiple network classes will have to be
    'networked' together to get a larger picture of the organism, with some
    subset of the morphogens ('nodes') strictly intracellular, cell-cell, or
    diffusive. The concentrations of morphogens not strictly intracellular
    will have to be intelligently mixed/updated according to the inter-network
    connections and diffusion coefficients before running the step function.
    Vohradsky 2001 Neural Model of the Genetic Network Gen. Prot. Bioinfor.

    This object doesn't actually store any of the morphogen concentrations.
    Those must be passed in as a matrix to the step/calculate functions.
    The idea is that one network is reused many times to simulate multiple
    cells with different morphogen concentration profiles.
    '''
    # max value for x in expression e^x
    EXP_MAX = math.floor(math.log(sys.float_info.max))

    def __init__(self,
                 n,
                 k1=None,
                 w=None,
                 b=None,
                 k2=None
                 ):
        '''
        var  description                size
        z   concentrations matrix @t    n
        k1  max expression rate         n
        w   directed weights matrix     n x n
        b   bias vector                 n
        k2  degradation rate            n
        '''
        # initalize internal neural network parameters
        self.n = n
        self.EXP_MAX_NUMPY = numpy.full(self.n, self.EXP_MAX)
        self.g = numpy.zeros(self.n, dtype=numpy.float64)
        self.dz = numpy.zeros(self.n, dtype=numpy.float64)
        self.edgeList = {}
        if k1 is None:
            self.k1 = numpy.array([1 for x in range(n)])
        else:
            self.k1 = numpy.array(k1)
        if w is None:
            self.w = numpy.array([[0 for x in range(n)] for y in range(n)])
        else:
            self.w = numpy.array(w)
        if b is None:
            self.b = numpy.array([1 for x in range(n)])
        else:
            self.b = numpy.array(b)
        if k2 is None:
            self.k2 = numpy.array([1 for x in range(n)])
        else:
            self.k2 = numpy.array(k2)

    def setWeight(self, i, j, wij):
        # set the weight with parents j, i
        # and maintain the edge list for fast looping
        # change weight of edge directed from node j to node i
        if self.w[i][j] == 0:
            if wij == 0:
                # no change
                pass
            else:
                # adding new weight
                self.edgeList.update([((i, j), wij)])
                pass
        elif wij == 0:
            # deleting weight
            del self.edgeList[(i, j)]
            pass
        else:
            # just changing the weight
            self.edgeList.update([((i, j), wij)])

        self.w[i][j] = wij
        # some possible mutations using this function:
        # • randomize proportional to current value. This modifies the strength
        # of intereaction one morphogen has on another only if they already
        # interact
        # • set weight to zero, breaking the interaction entierly
        # • setting small weights to zero, breaking weak iteractions
        # • set weight to some small number besides zero if it was already
        # zero, making a new interaction

        # probably would be much faster to store a list of edges
        # with a reference to their parent's index values and it's
        # weight

    def getWeight(self, i, j):
        return self.w[i][j]

    def addGene(self,
                k1=None,
                w=None,
                w2=None,
                b=None,
                k2=None,
                copy=None,
                newIndex=None):
        '''
        adds a node to the neural network (a gene/morphogen)
        all parameters are numbers, except w, which is a vector of length n
        (including the newly added node) which represents how the new node
        is affected by all other concentrations, and w2, which is a vector of
        length n-1 (not including the new node) which represents how every
        other node is affected by the new one. The last value in w represents
        the weight the new substance has on itself.
        '''

        if newIndex is None:
            newIndex = self.n

        self.n += 1
        self.EXP_MAX_NUMPY = numpy.full(self.n, self.EXP_MAX)
        self.g = numpy.zeros(self.n, dtype=numpy.float64)
        self.dz = numpy.zeros(self.n, dtype=numpy.float64)

        if copy is None:
            if k1 is not None:
                newParam = self.k1.tolist()
                newParam.insert(newIndex, k1)
                self.k1 = numpy.array(newParam)
            else:
                newk1 = self.k1.tolist()
                newk1.insert(newIndex, random.expovariate(1))
                self.k1 = numpy.array(newk1)
            if b is not None:
                newParam = self.b.tolist()
                newParam.insert(newIndex, b)
                self.b = numpy.array(newParam)
            else:
                newParam = self.b.tolist()
                newParam.insert(newIndex, random.gauss(0, 1))
                self.b = numpy.array(newParam)
            if k2 is not None:
                newParam = self.k2.tolist()
                newParam.insert(newIndex, k2)
                self.k2 = numpy.array(newParam)
            else:
                newParam = self.k2.tolist()
                newParam.insert(newIndex, random.expovariate(1))
                self.k2 = numpy.array(newParam)
        else:
            # copy parameters from copy, but not edges
            newParam = self.k2.tolist()
            newParam.insert(newIndex, self.k1[copy])
            self.k2 = numpy.array(newParam)
            newParam = self.b.tolist()
            newParam.insert(newIndex, self.b[copy])
            self.b = numpy.array(newParam)
            newParam = self.k2.tolist()
            newParam.insert(newIndex, self.k2[copy])
            self.k2 = numpy.array(newParam)

        # fill with zeros, no connections between the old nodes and the
        # new nodes, blank slate
        neww = self.w.tolist()
        for i in range(self.n-1):
            neww[i].insert(newIndex, 0)
        neww.insert(newIndex, [0 for i in range(self.n)])
        self.w = numpy.array(neww)

        # then modify w and weightList using setWeight() and the new data if
        # there is any
        if w is not None and w2 is not None:
            # first add an entry on the inputs to every other node
            for (i, w2i) in enumerate(w2):
                # self.w[i].append(w2i)
                self.setWeight(i, newIndex, w2i)
            # then add an entire row of inputs for the new node
            # self.w.append(w)
            for (j, wi) in enumerate(w):
                self.setWeight(newIndex, j, wi)

        # return index of new node
        return newIndex

    def removeGene(self, i):
        '''Remove gene i'''
        if self.n > 0:

            newParam = self.k1.tolist()
            newParam.pop(i)
            self.k1 = numpy.array(newParam)
            newParam = self.b.tolist()
            newParam.pop(i)
            self.b = numpy.array(newParam)
            newParam = self.k2.tolist()
            newParam.pop(i)
            self.k2 = numpy.array(newParam)

            # remove the input from gene i from all other nodes
            w = self.w.tolist()
            for jl in w:
                # modify the weight list for inputs from i
                jl.pop(i)

            # remove entire row on inputs for i
            w.pop(i)
            self.w = numpy.array(w)

            # entire edge list must be re-keyed since all indexes changed.
            newEdges = []
            for (parents, weight) in self.edgeList.items():
                newParent1 = parents[0]
                newParent2 = parents[1]
                # make sure this edge is not being deleted.
                if parents[0] != i and parents[1] != i:
                    if parents[0] > i:
                        # this parent was moved. The index is now one less
                        newParent1 -= 1
                    if parents[1] > i:
                        newParent2 -= 1

                    # only add the edge if it's not deleted
                    newEdges.append(((newParent1, newParent2),
                                     weight))

            self.n -= 1
            self.EXP_MAX_NUMPY = numpy.full(self.n, self.EXP_MAX)
            self.g = numpy.zeros(self.n, dtype=numpy.float64)
            self.dz = numpy.zeros(self.n, dtype=numpy.float64)
            # Update edgeList to contain the new keys
            self.edgeList = dict(newEdges)

    # now, some more organic network modification functions
    # split edge    create new node in place of an edge (insertNode)
    # flip edge
    # duplicate node
    # duplicated group of nodes (range of indexes)
    # change node index (regrouping/separating functional groups)
    # change group of nodes index (transposable elements)
    # move node along a connected edge ???
    # move group of nodes along a connected edge ???
    # delete node
    # delete group of nodes (range of indexes)
    # create random edge
    # delete random existing edge
    # redirect existing edge to random node
    # scale edge weight
    # negate weight
    # scale parameter (k1, b, k2)
    # negate bias

    def insertNode(self, edge=None, node=None):
        '''
        edge is a tuple (i, j) of parent indexes
        node is the index of the node to insert into the edge
        '''
        # this fuction might break if edge weight is 0

        if edge is None:
            edge = self.randomEdge()
            if edge is None:
                return
        if node is None:
            # create a new random node to insert
            node = self.addGene()

        # displace current edge with two new edges connected to node
        self.setWeight(edge[0][0], node, edge[1])
        self.setWeight(node, edge[0][1], edge[1])
        self.setWeight(edge[0][0], edge[0][1], 0)

    def flipEdge(self, edge=None):
        if edge is None:
            edge = self.randomEdge()
            if edge is None:
                return

        # flip the values of the reciprocal edges
        tmpWeight = self.getWeight(edge[0][1], edge[0][0])
        self.setWeight(edge[0][1], edge[0][0], edge[1])
        self.setWeight(edge[0][0], edge[0][1], tmpWeight)

    def duplicateNode(self, node=None, newNode=None):
        if node is None:
            node = self.randomNode()
            if node is None:
                return
        if newNode is None:
            # copy node
            newNode = self.addGene(copy=node)

        # copy all the incomming edges on node to newNode
        # except self nodes, convert those so they don't connect the clones
        # but instead connect the newNode to itself
        for (j, weight) in enumerate(self.w[node].copy()):
            if weight == 0:
                continue

            if j == node:
                # loopback weight, create a loopback weight on newNode
                self.setWeight(newNode, newNode, weight)
            else:
                # copy the weigts over, pointing to newNode
                self.setWeight(newNode, j, weight)
        # then copy the outgoing edges point from newNode
        # skip self edges
        for (i, weights) in enumerate(self.w.tolist()):
            if weights[node] == 0:
                # skip
                continue

            if i == node:
                # skip loopback edge
                continue

            self.setWeight(i, newNode, weights[node])

        # return new node
        return newNode

    def duplicateNodeGroup(self, nodeRange=None, meanLength=3):
        '''
        nodeRange is a tuple with the index of two nodes. Those nodes and
        all the noded between them will be duplicated
        '''
        # choose two random nodes. The set between these nodes will be
        # duplicated, including them. edges between nodes in the group
        # will be duplicated onto the new group. edges between nodes
        # within and nodes without the group will be duplicated to point
        # to/from the same outside nodes from/to the new node group
        if nodeRange is None:
            # choose random range, with a mean length
            r1 = random.randrange(self.n)
            length = int(random.gauss(0, meanLength))

            if length >= 0:
                if r1 + length < self.n:
                    # within range
                    nodeRange = (r1, r1 + length)
                else:
                    # go all the way to the end
                    nodeRange = (r1, self.n - 1)
            else:
                if r1 + length >= 0:
                    nodeRange = (r1 + length, r1)
                else:
                    # go to the end
                    nodeRange = (0, r1)
        else:
            # make sure the first index is smaller than or equal to the second
            if nodeRange[0] > nodeRange[1]:
                nodeRange = (nodeRange[1], nodeRange[0])

        copyRange = (self.n, self.n + nodeRange[1] - nodeRange[0])

        oldNodes = list(range(nodeRange[0], nodeRange[1] + 1))
        newNodes = list(range(copyRange[0], copyRange[1] + 1))

        # duplicate each node, with properties
        for node in oldNodes:
            self.addGene(copy=node)

        # Then duplicate edge structure
        for (index, oldNode) in enumerate(oldNodes):
            newNode = newNodes[index]

            # edges pointing to oldNode
            for (j, weight) in enumerate(self.w[oldNode].copy()):
                if weight == 0:
                    # skip
                    continue

                # check it the edge is coming from oldNode (loopback) inside
                # the group (nodeRange) or outside.
                if j == oldNode:
                    # loopback edge
                    self.setWeight(newNode, newNode, weight)
                elif j >= nodeRange[0] and j <= nodeRange[1]:
                    # edge comes from inside the group, so it should come from
                    # the corrisponding edge in the new group
                    newj = j - nodeRange[0] + copyRange[0]
                    self.setWeight(newNode, newj, weight)
                else:
                    # edge comes from outside the duplicated range.
                    self.setWeight(newNode, j, weight)

            # now duplicate edges pointing away from oldNode
            for (i, inputs) in enumerate(self.w.tolist()):
                weight = inputs[oldNode]
                if weight == 0:
                    # skip
                    continue

                # check it the edge is going to oldNode (loopback) inside
                # the group (nodeRange) or outside.
                if i == oldNode:
                    # loopback edge, skip
                    # self.setWeight(newNode, newNode, weight)
                    continue
                if i >= nodeRange[0] and i <= nodeRange[1]:
                    # edge points inside the group, so it should point to
                    # the corrisponding edge in the new group
                    newi = i - nodeRange[0] + copyRange[0]
                    self.setWeight(newi, newNode, weight)
                else:
                    # edge points outside the duplicated range.
                    self.setWeight(i, newNode, weight)

        # returns the range of old nodes
        # new nodes are appended, same length as old nodes
        return nodeRange

    def changeNodeIndex(self, oldIndex=None, newIndex=None):
        ''' Move a node. Returns an updated position of the newIndex'''
        if oldIndex is None:
            oldIndex = random.randrange(self.n)
        if newIndex is None:
            newIndex = random.randrange(self.n)

        # duplicate parameters into custom index
        newNode = self.addGene(copy=oldIndex, newIndex=newIndex)

        # old index may be shifted by the new insertion
        if oldIndex >= newIndex:
            oldIndex += 1

        # duplicate connections onto copy
        self.duplicateNode(oldIndex, newNode)

        # remove old node
        self.removeGene(oldIndex)

        # new index may be shifted
        if newIndex > oldIndex:
            newIndex -= 1

        # return the updated index position of the new node
        return newIndex

    def randomNode(self):
        if self.n == 0:
            return None
            # random.randrange(
        ret = random.randrange(self.n)
        return ret

    def randomEdge(self):
        # choose random edge
        edges = list(self.edgeList.items())
        if len(edges) == 0:
            # if there are no edges, stop
            return None
        return edges[random.randrange(len(edges))]

    def f(self, x):
        '''Sigmoidal function'''
        # make sure to avoid Overflow errors, clamp to zero
        if -x > self.EXP_MAX:
            return 0

        # Transfer function, sigmoidal function
        # other transfer functions might be more performant, but this seems
        # decently fast
        return 1 / (1 + math.exp(-x))

    def calculate_dz(self, z):
        '''returns the change in concentration for each morphogen.
        z is a numpy.array(self.n, dtype=numpy.float64) object
        returns the same kind of object'''
        # calculate the rate of change of each morphogen
        # according to the stored weights, bias, rate constants.
        # useful when there is an underlying concentration map that should
        # be shared between many Network objects. Allows some external code
        # to choose how to update which concentrations of what Networks
        # given the calculated dz's from each of them. This is the
        # multi-cellular case
        # dzi = k1i * f( SUM over j wij * zj + bi ) - k2i * zi

        # calculate the expression rates based on all concentrations and
        # connection weights
        # initialize to bias vector
        numpy.copyto(self.g, self.b)
        # for each edge, calculate the sum of effect it has on each output i
        # using the more efficient edgeList dictionary
        for (parents, weight) in self.edgeList.items():
            # for each input node j, sum the effect it has on output i
            self.g[parents[0]] += weight * z[parents[1]]

        # make sure to avoid Overflow errors, clamp to zero
        self.g *= -1
        # avoid allocating new memory by using in place operations
        numpy.exp(numpy.minimum(self.g, self.EXP_MAX_NUMPY), out=self.dz)
        self.dz += 1
        self.dz **= -1
        # numpy.reciprocal(out, out=out)
        self.dz *= self.k1
        self.dz -= self.k2 * z

        return self.dz

    def step(self, z, dt):
        '''z is a numpy.array(self.n, dtype=numpy.float64)'''
        # do the integration on the self stored concentration matrix.
        # only useful if the Network object isn't interacting with other
        # Network objects.

        dz = self.calculate_dz(z)

        # update the concentrations based on the expression rates
        # this function is mostly for testing. Usually you'd want a more
        # complex relationship between the mophogen concentrations in different
        # matrixes, not just updating the concentrations of a single matrix
        # 1-to-1
        dz *= dt
        z += dz


def osscilatingCircuit():
    # testing the cell system manually
    # The starting concentration of each morphogen, very tiny noise to randomize
    n = 3
    z = [random.random() / 1000000 for i in range(n)]
    w = [[0, -1000, 0],
         [0, 0, -1000],
         [-1000, 0, 0]]
# w = [[random.gauss(mu=0, sigma=500) for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            if random.random() < 0.50:
                # w[i][j] = 0
                pass
    k1 = [abs(random.gauss(1, 1)) for i in range(n)]
    b = [abs(random.gauss(3, 1)) for i in range(n)]
    k2 = [abs(random.gauss(2, 1)) for i in range(n)]
    k1 = [1 for i in range(n)]
    b = [1 for i in range(n)]
    k2 = [1 for i in range(n)]
    pprint.pprint(w)
    print("..")
    pprint.pprint([k1, b, k2])
    cell = RegulatoryNetwork(n,
                             w=w,
                             k1=k1,
                             b=b,
                             k2=k2)

    while True:
        string = ''
        for morphogen in z:
            string += '->|' + '='*int(morphogen * 500) + '\n'
        print(string)
        cell.step(z, 0.01)
        # delay
        for i in range(1000000):
            pass


# osscilatingCircuit()


class CellNetwork:
    '''
    Defines a population of cells connected in a network. Defines which
    morphogens are intra-cell, cell-cell direct contact, or diffusive, and
    calculates the morphogen concentration for the whole network for each
    timestep by calling the Cell.calculate_dz() function on each cell, summing
    up the appropriate dz's, and applying a z(t+dt) = z(t) + dz*dt to each
    concentration in each cell, in addition to the diffusion kernel.

    basically, every cell has all it's concentrations, we just apply dz from
    each cell to either it's own z's, it's neighbors z's, or some combination
    according to the transport process involved. Each cell is mixed, but
    transport betwen cells is controlled by this class

    All cells have the same regulatory network inside, they are only
    differentiated by the concentrations of morphogens present in each cell.
    '''

    def __init__(self):
        # let's just make a grid for testing
        pass

    pass
