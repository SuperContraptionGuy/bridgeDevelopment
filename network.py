import math
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
                 edgeList=None,
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
        if k1 is None:
            self.k1 = [1 for x in range(n)]
        else:
            self.k1 = k1
        if edgeList is None:
            self.edgeList = {}
        else:
            self.edgeList = edgeList
        if b is None:
            self.b = [1 for x in range(n)]
        else:
            self.b = b
        if k2 is None:
            self.k2 = [1 for x in range(n)]
        else:
            self.k2 = k2

    def setWeight(self, i, j, wij):
        # set the weight with parents j, i
        # and maintain the edge list for fast looping
        # change weight of edge directed from node j to node i
        if (i, j) in self.edgeList:
            if wij == 0:
                # delete the existing edge
                del self.edgeList[(i, j)]
            else:
                # change the existing edge
                self.edgeList[(i, j)] = wij
        else:
            if wij != 0:
                # add a new edge
                self.edgeList[(i, j)] = wij
            else:
                # don't add empty edges
                pass

        # some possible mutations using this function:
        # • randomize proportional to current value. This modifies the strength
        # of intereaction one morphogen has on another only if they already
        # interact
        # • set weight to zero, breaking the interaction entierly
        # • setting small weights to zero, breaking weak iteractions
        # • set weight to some small number besides zero if it was already
        # zero, making a new interaction

    def getWeight(self, i, j):
        if (i, j) in self.edgeList:
            # return the value of existing edge
            return self.edgeList[(i, j)]
        else:
            # return 0, the weight of non-edges
            return 0

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

        if newIndex is not None:
            # with a newIndex that is not self.n the edge list will be outdated
            # offset any index in the edge list if it is equal to or greater
            # than newIndex
            newEdgeList = {}
            for (parents, weight) in self.edgeList.items():
                newParents = parents
                if parents[0] >= newIndex and parents[1] >= newIndex:
                    # both parents are larger than new index
                    newParents = (parents[0] + 1, parents[1] + 1)
                elif parents[0] >= newIndex:
                    # first parent is larger
                    newParents = (parents[0] + 1, parents[1])
                elif parents[1] >= newIndex:
                    # second parent is larger
                    newParents = (parents[0], parents[1] + 1)
                # add edge
                newEdgeList[newParents] = weight
            # update the edge list
            self.edgeList = newEdgeList

        else:
            # otherwise, append the new node to the end. edges are preserved
            newIndex = self.n

        self.n += 1

        if copy is None:
            if k1 is not None:
                self.k1.insert(newIndex, k1)
            else:
                self.k1.insert(newIndex, random.expovariate(1))
            if b is not None:
                self.b.insert(newIndex, b)
            else:
                self.b.insert(newIndex, random.gauss(0, 1))
            if k2 is not None:
                self.k2.insert(newIndex, k2)
            else:
                self.k2.insert(newIndex, random.expovariate(1))
        else:
            # copy parameters from copy, but not edges
            self.k1.insert(newIndex, self.k1[copy])
            self.b.insert(newIndex, self.b[copy])
            self.k2.insert(newIndex, self.k2[copy])

        # then modify w and weightList using setWeight() and the new data if
        # there is any
        if w is not None and w2 is not None:
            # first add an entry on the inputs to every other node
            for (i, w2i) in enumerate(w2):
                self.setWeight(i, newIndex, w2i)
            # then add an entire row of inputs for the new node
            for (j, wi) in enumerate(w):
                self.setWeight(newIndex, j, wi)

        # return index of new node
        return newIndex

    def removeGene(self, i):
        '''Remove gene i'''
        if self.n > 0:

            self.k1.pop(i)
            self.b.pop(i)
            self.k2.pop(i)

            # entire edge list must be re-keyed since all indexes changed.
            # edges that involve the node being deleted must also be deleted
            newEdgeList = {}
            for (parents, weight) in self.edgeList.items():
                newParents = parents
                if parents[0] == i or parents[1] == i:
                    # this edge must not be copied over, it involves to node
                    # being deleted.
                    continue
                elif parents[0] > i and parents[1] > i:
                    # both parents are larger than new index
                    newParents = (parents[0] - 1, parents[1] - 1)
                elif parents[0] > i:
                    # first parent is larger
                    newParents = (parents[0] - 1, parents[1])
                elif parents[1] > i:
                    # second parent is larger
                    newParents = (parents[0], parents[1] - 1)
                # add edge
                newEdgeList[newParents] = weight
            # update the edge list
            self.edgeList = newEdgeList

            self.n -= 1

    # TODO: modify these functions so the index of the new nodes is related to
    #   the derivitive nodes
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
                # there are no nodes to copy
                return
        if newNode is None:
            # copy node,
            # TODO: with an index adjacent to the old node, without breaking
            #   edge duplication code
            newNode = self.addGene(copy=node)

        # copy all the incomming edges on node to newNode
        # except self nodes, convert those so they don't connect the clones
        # but instead connect the newNode to itself
        for (parents, weight) in self.edgeList.copy().items():
            if parents[0] == node and parents[1] == node:
                # loop back weight
                self.setWeight(newNode, newNode, weight)
            elif parents[0] == node:
                # pointed at old node, so point an edge at new node
                self.setWeight(newNode, parents[1], weight)
            elif parents[1] == node:
                # pointed from old node, so point an edge from new node
                self.setWeight(parents[0], newNode, weight)
            else:
                # not related to node being copied, so don't bother it
                pass

        # return new node
        return newNode

    def duplicateNodeGroup(self, nodeRange=None, newIndex=None, meanLength=3):
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

        if newIndex is None:
            # can be any index in self.n, including self.n (after everything)
            newIndex = random.randrange(self.n + 1)

        copyRange = (newIndex, newIndex + nodeRange[1] - nodeRange[0])

        copyLength = nodeRange[1] - nodeRange[0] + 1
        if copyRange[0] <= nodeRange[0]:
            # new nodes entirely ahead of old ones, offset entire old range
            oldNodes = list(range(nodeRange[0] + copyLength,
                                  nodeRange[1] + copyLength + 1))
        elif copyRange[0] <= nodeRange[1]:
            # new nodes are nested within old range, offset some of them
            oldNodes = list(range(nodeRange[0],
                                  copyRange[0]))
            oldNodes.extend(list(range(copyRange[1] + 1,
                                       nodeRange[1] + copyLength + 1)))
        else:
            # new range is after old one, so no offset occurs
            oldNodes = list(range(nodeRange[0], nodeRange[1] + 1))
        newNodes = list(range(copyRange[0], copyRange[1] + 1))

        # duplicate each node, with properties
        for (index, node) in enumerate(oldNodes):
            self.addGene(copy=node,
                         newIndex=newNodes[index])

        # now create the copies. the results after should be in the positions
        # specified by oldNodes and newNodes
        oldNodeIndex = nodeRange[0]
        newNodeIndex = copyRange[0]
        for i in range(nodeRange[0], nodeRange[1] + 1):
            self.addGene(copy=oldNodeIndex, newIndex=newNodeIndex)
            oldNodeIndex += 1
            newNodeIndex += 1
            if oldNodeIndex == copyRange[0]:
                # In this case, the new range is right in the middle
                # of the old range, so skip to the end of the range (i + 1)
                # and add twice for every iteration afterwards (next if)
                oldNodeIndex += i + 1
            if oldNodeIndex > copyRange[0]:
                # if old index is after the new one, it will offset twice
                # every iteration, so here's a second addition
                oldNodeIndex += 1

        # classify every edge
        #   inter group edges: duplicate onto new group
        #   group -> external: duplicate newgroup -> external
        #   external -> group: duplicate external -> newgroup
        #   external -> external: do nothing, skip
        # and duplicate the edge structure accordingly
        for (parents, weight) in self.edgeList.copy().items():

            parent0_internal = parents[0] >= nodeRange[0] and\
                               parents[0] <= nodeRange[1]
            parent1_internal = parents[1] >= nodeRange[0] and\
                               parents[1] <= nodeRange[1]
            # these are valid if corrisponding parent is in-group
            # maps the old group indexes to their duplicates in the new group
            newi = parents[0] - nodeRange[0] + copyRange[0]
            newj = parents[1] - nodeRange[0] + copyRange[0]

            if not parent0_internal and not parent1_internal:
                # edge is external -> external, skip
                continue
            if parent0_internal and parent1_internal:
                # edge is internal -> internal, intergroup
                # duplicate onto new group
                self.setWeight(newi, newj, weight)
            elif parent0_internal:
                # group -> external
                self.setWeight(newi, parents[1], weight)
            elif parent1_internal:
                # external -> group
                self.setWeight(parents[0], newj, weight)

        # returns the range of old nodes
        # new nodes are appended, same length as old nodes
        return nodeRange

    def changeNodeIndex(self, oldIndex=None, newIndex=None):
        ''' Move a node. Returns an updated position of the newIndex'''
        if oldIndex is None:
            oldIndex = random.randrange(self.n)
        if newIndex is None:
            newIndex = random.randrange(self.n)

        if newIndex > oldIndex:
            # if the new index will be offset when we delete the old node,
            # shift it by one.
            newIndex += 1

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
        if self.n != len(z):
            print("NOT EQUAL: ", self.n, len(z))
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
        g = [self.b[x] for x in range(self.n)]
        # for each edge, calculate the sum of effect it has on each output i
        # using the more efficient edgeList dictionary
        for (parents, weight) in self.edgeList.items():
            # for each input node j, sum the effect it has on output i
            g[parents[0]] += weight * z[parents[1]]

        dz = [0 for x in range(self.n)]
        # for each output node i, calculate rate of change of concentration
        for (i, zi) in enumerate(z):
            dz[i] = self.k1[i] * self.f(g[i]) - self.k2[i] * zi

        return dz

    def step(self, z, dt):
        # do the integration on the self stored concentration matrix.
        # only useful if the Network object isn't interacting with other
        # Network objects.

        dz = self.calculate_dz(z)

        # update the concentrations based on the expression rates
        # this function is mostly for testing. Usually you'd want a more
        # complex relationship between the mophogen concentrations in different
        # matrixes, not just updating the concentrations of a single matrix
        # 1-to-1
        for (i, dzi) in enumerate(dz):
            z[i] += dzi * dt


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
