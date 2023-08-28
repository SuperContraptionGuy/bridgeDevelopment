import math
import random
import sys
import pygame.color
import vecUtils as v

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
    node_r = 20

    def __init__(self,
                 addConcentrationFunc=None,
                 removeConcentrationFunc=None
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

        # per cell variables
        self.z = []

        # general shared parameters
        self.addConcentrationFunc = addConcentrationFunc
        self.removeConcentrationFunc = removeConcentrationFunc

        self.k1 = []        # max expression rate
        self.k2 = []        # degredation rate constant
        self.b = []         # bias
        self.sigma = []     # gaussian diffusion coefficient

        self.n = 0          # number of morphogens

        self.edgeList = {}  # regulatory iteraction weights

        # display parameters, also shared
        self.locs = []
        self.colors = []
        self.displayIndicators = []

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
                loc=None,
                color=None,
                k1=None,
                w=None,
                w2=None,
                b=None,
                k2=None,
                sigma=None,
                copy=None,
                newIndex=None):
        '''
        adds a node to the neural network (a gene/morphogen)

        if copy is provided, the parameters are copied from that node, but
        over ridden if also explicitly provided in the arguments.

        add a new gene and associated interface properties.
        loc is the coordinate position of the node in the editor
            if not provided, a random loc is chosen around the origin

        if newIndex is provided, then that will the the index of the new
        node. newIndex must be from 0 through self.n.
        Otherwise it's inserted randomly

        To add weights, w is a vector of length n
        (including the newly added node) which represents how the new node
        is affected by all other concentrations, and w2, which is a vector of
        length n-1 (not including the new node) which represents how every
        other node is affected by the new one. The last value in w represents
        the weight the new substance has on itself.

        '''

        if newIndex is None:
            # put the node somewhere randomly
            # newIndex = random.randrange(self.n + 1)
            # put it at the end
            # newIndex = self.n
            newIndex = random.randrange(self.n + 1)

        # for display at least, if not simulation
        self.z.insert(newIndex, random.expovariate(1))
        if self.addConcentrationFunc is not None:
            # use an external function to manage concentrations
            self.addConcentrationFunc(newIndex)

        self.n += 1
        self.displayIndicators.insert(newIndex, False)

        if copy is not None:
            # copy parameters from copy, but not edges
            self.k1.insert(newIndex, self.k1[copy])
            self.b.insert(newIndex, self.b[copy])
            self.k2.insert(newIndex, self.k2[copy])
            self.sigma.insert(newIndex, self.sigma[copy])
            self.locs.insert(newIndex, self.locs[copy])
            self.colors.insert(newIndex, pygame.Color(self.colors[copy]))

        else:
            # populate with default values
            self.k1.insert(newIndex, random.expovariate(1))
            self.b.insert(newIndex, random.gauss(0, 1))
            self.k2.insert(newIndex, random.expovariate(1))
            self.sigma.insert(newIndex, random.expovariate(1))
            # self.locs.insert(newIndex, (0, 0))
            # use a random location
            self.locs.insert(newIndex,
                             v.mul(v.u(random.uniform(0, math.pi * 2)),
                                   random.gauss(0, 200)))
            # choose random color for the new gene
            color = pygame.Color((0, 0, 0))
            color.hsva = (random.uniform(0, 360), 80, 50)
            self.colors.insert(newIndex, color)

        # replace copied properties if they are explicitly specified
        if k1 is not None:
            self.k1[newIndex] = k1
        if b is not None:
            self.b[newIndex] = b
        if k2 is not None:
            self.k2[newIndex] = k2
        if sigma is not None:
            self.sigma[newIndex] = sigma
        if loc is not None:
            self.locs[newIndex] = loc
        if color is not None:
            self.colors[newIndex] = color

        # rekey edge list
        newEdgeList = {}
        for (parents, weight) in self.edgeList.items():
            newParents = parents
            if parents[0] >= newIndex and parents[1] >= newIndex:
                # both parents are now larger than new index
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

    def removeGene(self, i=None, removeConcentrationFunc=None):
        '''
        Remove gene i and recalculates the edge list to match the shifted
        indicies
        '''
        if self.n == 0:
            return
        if i is None:
            i = self.randomNode()
            if i is None:
                return

        # for display at least
        self.z.pop(i)
        if self.removeConcentrationFunc is not None:
            # use an external function to manage concentrations
            self.removeConcentrationFunc(i)

        self.k1.pop(i)
        self.b.pop(i)
        self.k2.pop(i)
        self.sigma.pop(i)
        self.locs.pop(i)
        self.colors.pop(i)
        self.displayIndicators.pop(i)

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

    def removeGeneGroup(self, nodeRange=None):
        '''
        Removes a random range of genes or that given by nodeRange
        which is an inclusive tuple
        (start, last)
        Edge list is recalculated
        '''
        if self.n < 1:
            return

        if nodeRange is None:
            nodeRange = self.randomNodeRange()

        index = nodeRange[0]
        for node in range(nodeRange[0], nodeRange[1] + 1):
            # the lazy way. more efficient way would be to only recalculate
            # edges after all genes are deleted. right now recalculation
            # is bundled with removeGene, so is ran every time
            self.removeGene(index)

    # TODO: modify these functions so the index of the new nodes is related to
    #   the derivitive nodes
    # now, some more organic network modification functions

    # add random gene
    # delete node
    # delete group of nodes (range of indexes)
    # split edge    create new node in place of an edge (insertNode)
    # flip edge
    # duplicate node
    # duplicated group of nodes (range of indexes)
    # change node index (regrouping/separating functional groups)
    # change group of nodes index (transposable elements)
    # create random edge
    # delete random existing edge
    # scale existing edge weight
    # negate weight
    # redirect existing edge to random node
    # scale parameter (k1, b, k2)
    # negate bias

    # move node along a connected edge ???
    # move group of nodes along a connected edge ???

    # there are complemetary mutations that keep the number of nodes, and some
    #   without complements
    # and edges on a random walk
    # addGene - deleteGene
    # split edge - deleteGene
    # flip edge
    # duplicate node - delete node
    # duplicate group - delete group
    # change index
    # change group index
    # create random edge - delete random edge (roughly even. delete advantage)
    # scale existing edge weight
    # negate weight
    # redirect existing edge (slightly uneven, sometimes deletes edges)
    # scale parameters
    # negate bias

    # maybe adjust the random node and randomrange functions to allow
    #   weighted distrubutions around a point. The parameters for these
    #   distributions could be stored per gene.

    def splitEdge(self, edge=None, node=None, newIndex=None):
        '''
        Inserts a node inbetween the two parents of edge, replacing the
        original edge with two edges.
        Eg.
        Parent1 <- Parent2
        Parent1 <- insertedNode <- Parent2

        If edge is not provided, a random edge is chosen
        edge is tuple, with structure like (parents, weight)
        and parents is also a tuple like (parent1, parent2)

        If newIndex is provided, that's where the new node will be inserted,
        otherwise the new node will be inserted between the original parents,
        or before/after a loopback edge.

        If node is provided, than this existing node will be used, and
        the edges will be connected to it.
        '''
        # this fuction might break if edge weight is 0

        if edge is None:
            edge = self.randomEdge()
            if edge is None:
                return
        if newIndex is None:
            # select an index between the parent nodes of the edge
            # or before/after if it's a loopback edge
            # newIndex = int(math.ceil((edge[0][0] + edge[0][1]) / 2))
            if edge[0][0] < edge[0][1]:
                newIndex = random.randrange(edge[0][0] + 1, edge[0][1] + 1)
            elif edge[0][0] > edge[0][1]:
                newIndex = random.randrange(edge[0][1] + 1, edge[0][0] + 1)
            else:
                # this is a loopback edge, insert before or after
                newIndex = random.randrange(edge[0][0], edge[0][0] + 1)

        if node is None:
            # create a new random node to insert, put it at the end, move
            # later
            node = self.addGene(newIndex=self.n)

        # displace current edge with two new edges connected to node
        self.setWeight(edge[0][0], node, edge[1])
        self.setWeight(node, edge[0][1], edge[1])
        self.setWeight(edge[0][0], edge[0][1], 0)

        if edge[0][0] == edge[0][1]:
            # this is a loopback edge, offset the new node
            newNodePos = v.sum(self.locs[edge[0][0]],
                               v.mul(v.u(random.uniform(0, math.pi * 2)),
                                     self.node_r * 3))
        else:
            # this is a standard edge, make position average of parents
            newNodePos = v.div(v.sum(self.locs[edge[0][0]],
                                     self.locs[edge[0][1]]), 2)
        # mix the colors of the two parent nodes
        hue1 = self.colors[edge[0][0]].hsva[0]
        hue2 = self.colors[edge[0][1]].hsva[0]
        newColor = pygame.Color(self.colors[edge[0][0]])
        # take the average hue, making sure to use the shortest circular
        # distance between the two hues
        if abs(hue2 - hue1) <= 180:
            newHue = (hue1 + hue2) / 2
        else:
            newHue = ((hue1 + hue2) / 2 + 180) % 360
        newColor.hsva = (newHue,
                         newColor.hsva[1],
                         newColor.hsva[2],
                         newColor.hsva[3])

        self.locs[node] = newNodePos
        self.colors[node] = newColor
        # change node index to new index
        if newIndex != node:
            self.changeNodeIndex(node, newIndex)

    def flipEdge(self, edge=None):
        '''
        flips the direction of edge.

        If edge is not provided, a random edge is chosen

        edge is tuple, with structure like (parents, weight)
        and parents is also a tuple like (parent1, parent2)
        '''
        if edge is None:
            edge = self.randomEdge()
            if edge is None:
                return

        # flip the values of the reciprocal edges
        tmpWeight = self.getWeight(edge[0][1], edge[0][0])
        self.setWeight(edge[0][1], edge[0][0], edge[1])
        self.setWeight(edge[0][0], edge[0][1], tmpWeight)

    def duplicateNode(self, node=None, newNode=None, newIndex=None):
        '''
        Duplicates a single node with index node to newIndex.

        If newIndex is not provided, then the duplicate will be inserted
        either directly ahead or directly after node

        If newNode is provided, than a new node is not created, but instead
        the edge structure on node is duplicated onto newNode.
        '''
        if node is None:
            node = self.randomNode()
            if node is None:
                # there are no nodes to copy
                return
        if newIndex is None:
            # if no newIndex is given, add the duplicate node right next to
            # original node, either before or after it
            newIndex = node + random.randrange(2)

        if newNode is None:
            # copy node,
            newNode = self.addGene(copy=node, newIndex=newIndex)
            if newNode <= node:
                # the newGene offset the original, so adjust node's index to
                # point to the actual old node
                node += 1

        # position the new node randomly offset from old one
        newNodePos = v.sum(self.locs[node],
                           v.mul(v.u(random.uniform(0, math.pi * 2)),
                                 self.node_r * 1.5))
        # generate a color offset from old node
        newColor = pygame.Color(self.colors[node])
        newHue = (newColor.hsva[0] +
                  random.vonmisesvariate(0, 8) * 360 / 2 / math.pi) % 360
        newColor.hsva = (newHue,
                         newColor.hsva[1],
                         newColor.hsva[2],
                         newColor.hsva[3])
        self.locs[newNode] = newNodePos
        self.colors[newNode] = newColor

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
        Duplicates a range of nodes from nodeRange[0] through nodeRange[1]
        and inserts them starting at newIndex.

        If nodeRange is not provided, then a random range with a mean length
        of meanLength will be chosen

        If newIndex is not provided, then the duplicated range will be inserted
        either directly ahead or directly behind nodeRange
        '''
        # choose two random nodes. The set between these nodes will be
        # duplicated, including them. edges between nodes in the group
        # will be duplicated onto the new group. edges between nodes
        # within and nodes without the group will be duplicated to point
        # to/from the same outside nodes from/to the new node group
        if self.n == 0:
            # can't do anything with no nodes
            return (None, None, None)
        if nodeRange is None:
            nodeRange = self.randomNodeRange(meanLength)
        else:
            # make sure the first index is smaller than or equal to the second
            if nodeRange[0] > nodeRange[1]:
                nodeRange = (nodeRange[1], nodeRange[0])

        if newIndex is None:
            # can be any index in self.n, including self.n (after everything)
            # newIndex = random.randrange(self.n + 1)
            # set the newIndex to be adjacent to nodeRange
            newIndex = random.choice([nodeRange[0], nodeRange[1] + 1])

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
        originalOldNodes = list(range(nodeRange[0], nodeRange[1] + 1))
        newNodes = list(range(copyRange[0], copyRange[1] + 1))

        # calculate extent of node group to allow offset
        # measure distance between all nodes in group
        # store largest distance, as a vector
        # I don't like this distance calculator sometimes it gets big
        maxDist = self.node_r * 2
        maxDistVec = v.mul(v.u(random.vonmisesvariate(0, 0)), maxDist)
        for node1 in originalOldNodes:
            for node2 in originalOldNodes:
                vec = v.sub(self.locs[node1], self.locs[node2])
                dist = v.mag(vec)
                if dist > maxDist:
                    maxDist = dist
                    maxDistVec = vec
        # offset all the nodes in the new group by an amount perpendicular to
        # distvec
        offset = v.perp(v.mul(maxDistVec, 0.5))
        hueOffset = random.vonmisesvariate(0, 8) * 360 / 2 / math.pi

        # now create the copies. the results after should be in the positions
        # specified by oldNodes and newNodes
        oldNodeIndex = nodeRange[0]
        newNodeIndex = copyRange[0]
        for i in range(copyLength):
            # generate a color offset from the corrisponding old node
            newColor = pygame.Color(self.colors[oldNodeIndex])
            newHue = (newColor.hsva[0] + hueOffset) % 360
            newColor.hsva = (newHue,
                             newColor.hsva[1],
                             newColor.hsva[2],
                             newColor.hsva[3])
            self.addGene(loc=v.sum(self.locs[oldNodeIndex],
                                   offset),
                         color=newColor,
                         copy=oldNodeIndex,
                         newIndex=newNodeIndex)
            oldNodeIndex += 1
            newNodeIndex += 1
            if oldNodeIndex == copyRange[0]:
                # In this case, the new range is right in the middle
                # of the old range, so skip to the end of the range (i + 1)
                # and add twice for every iteration afterwards (next if)
                oldNodeIndex += i + 1
            elif oldNodeIndex > copyRange[0]:
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

            # parent0_internal = parents[0] >= nodeRange[0] and\
            #                    parents[0] <= nodeRange[1]
            # parent1_internal = parents[1] >= nodeRange[0] and\
            #                    parents[1] <= nodeRange[1]
            parent0_internal = parents[0] in oldNodes
            parent1_internal = parents[1] in oldNodes

            # these are valid if corrisponding parent is in-group
            # maps the old group indexes to their duplicates in the new group
            # newi = parents[0] - nodeRange[0] + copyRange[0]
            # newj = parents[1] - nodeRange[0] + copyRange[0]
            if parent0_internal:
                newi = newNodes[oldNodes.index(parents[0])]
            if parent1_internal:
                newj = newNodes[oldNodes.index(parents[1])]

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

        # returns the list of original oldnodes,
        # old nodes and new nodes, mapped
        return (originalOldNodes, oldNodes, newNodes)

    def changeNodeIndex(self, oldIndex=None, newIndex=None):
        '''
        Move a node. Returns an updated position of the newIndex

        Change the index of node at oldIndex to newIndex.
        newIndex max value is self.n - 1
        '''
        if self.n == 0:
            return
        if oldIndex is None:
            oldIndex = self.randomNode()
        if newIndex is None:
            newIndex = self.randomNode()

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

    def changeNodeGroupIndex(self, nodeRange=None, newIndex=None):
        '''
        changes index of nodes in nodeRange to start from newIndex.

        Move a group of node represented by nodeRange to a new index position
        newIndex has a range from 0 to self.n - (nodeRange[1] - nodeRange[0]
        '''
        if self.n < 1:
            return

        if nodeRange is None:
            # choose random range
            nodeRange = self.randomNodeRange()
        if newIndex is None:
            # choose a random index, not including the nodeRange
            copyLength = nodeRange[1] - nodeRange[0] + 1
            newIndex = random.randrange(self.n - copyLength + 1)
        if newIndex > nodeRange[0]:
            # new index should be offset past nodeRange, so that when nodeRange
            # is deleted, the index of  the duplicated range starts at the
            # original newIndex
            newIndex += nodeRange[1] - nodeRange[0] + 1
            # because of this, newIndex cannot be greater than
            # self.n - lengthOfnodeRange

        (originalOldNodes, oldNodes, newNodes) = \
            self.duplicateNodeGroup(nodeRange, newIndex)

        oldNodeIndex = oldNodes[0]
        copyLength = newNodes[-1] - newNodes[0] + 1
        for i in range(copyLength):
            self.removeGene(oldNodeIndex)
            if oldNodeIndex == newNodes[0]:
                # In this case, the new range is right in the middle
                # of the old range, so skip to the end of the range
                oldNodeIndex += copyLength

    def addEdge(self, i=None, j=None, weight=None):
        '''
        add a random edge to the graph with random weight.

        Does not override existing edges

        If provided, i, j, and weight define parameters of the new edge.
        The new edge points from j to i with weight weight.

        if any parameter is not provided, it is randomized
        '''
        # if not provided, randomize properites of the new edge
        if self.n < 1:
            # no nodes, exit
            return
        if i is None:
            i = self.randomNode()
        if j is None:
            j = self.randomNode()
        if weight is None:
            weight = random.gauss(0, 1)

        if (i, j) in self.edgeList:
            # this edge already exists
            return

        self.setWeight(i, j, weight)

    def removeEdge(self, i=None, j=None):
        '''
        remove a random existing edge

        If provided, i and j define parameters of the new edge.
        The new edge points from j to i.

        if any parameter is not provided, it is randomized

        if neither i or j are provided, then a random existing edge is deleted
        '''
        # if not provided, randomize properites of the new edge
        if len(self.edgeList) == 0:
            # no edges, exit
            return
        if i is not None or j is not None:
            if i is None:
                i = self.randomNode()
            if j is None:
                j = self.randomNode()

        else:
            edge = self.randomEdge()
            i = edge[0][0]
            j = edge[0][1]

        self.setWeight(i, j, 0)

    def scaleWeight(self, i=None, j=None, scaler=None):
        '''
        Scales a weight by a constant factor.

        If no weight is specified, it scales a random weight.

        if no scaler is specified, it chooses a random scaling factor,
        always positive and mean of 1
        '''
        if len(self.edgeList) == 0:
            # no edges, exit
            return
        if i is not None or j is not None:
            if i is None:
                i = self.randomNode()
            if j is None:
                j = self.randomNode()

        else:
            edge = self.randomEdge()
            i = edge[0][0]
            j = edge[0][1]

        if scaler is None:
            scaler = random.gammavariate(4, 1 / 4)

        self.setWeight(i, j, self.getWeight(i, j) * scaler)

    def negateWeight(self, i=None, j=None):
        '''
        Negates a weight. if none specified, negates a random weight
        '''
        if len(self.edgeList) == 0:
            # no edges, exit
            return
        if i is not None or j is not None:
            if i is None:
                i = self.randomNode()
            if j is None:
                j = self.randomNode()

        else:
            edge = self.randomEdge()
            i = edge[0][0]
            j = edge[0][1]

        self.setWeight(i, j, self.getWeight(i, j) * -1)

    def redirectEdge(self, i=None, j=None, newi=None, newj=None):
        '''
        Redirects an edge from (i, j) to (newi, newj).

        if any parameters are missing, they are randomized.

        if no parameters are supplied, one of the two starting parents
        will be perserved
        '''
        if len(self.edgeList) == 0:
            # no edges, exit
            return
        if i is not None or j is not None:
            if i is None:
                i = self.randomNode()
            if j is None:
                j = self.randomNode()

            weight = self.getWeight(i, j)

        else:
            # pick a random edge
            edge = self.randomEdge()
            i = edge[0][0]
            j = edge[0][1]
            weight = edge[1]

        if newi is not None or newj is not None:
            if newi is None:
                newi = self.randomNode()
            if newj is None:
                newj = self.randomNode()

        else:
            # choose whether to redirect the tip or tail
            if random.choice([True, False]):
                # tip
                newi = self.randomNode()
                newj = j
            else:
                # tail
                newi = i
                newj = self.randomNode()

        self.setWeight(i, j, 0)
        self.setWeight(newi, newj, weight)

    def scaleParameter(self, node=None, k1=None, k2=None, b=None, sigma=None):
        '''
        scales a parameter of node.

        If no parameters after node are given, one parameter is randomly chosen
        and scaled a random amount. (scale factor always positive, mean of 1)

        if no node given, a random node is chosen

        if given, takes k1, k2, and/or b as scaling factors to adjust the
        existing weights of node.
        '''
        if self.n < 1:
            return

        if node is None:
            node = self.randomNode()
        if k1 is not None or k2 is not None or b is not None:
            if k1 is None:
                k1 = random.gammavariate(4, 1 / 4)
            if k2 is None:
                k2 = random.gammavariate(4, 1 / 4)
            if b is None:
                b = random.gammavariate(4, 1 / 4)
            if sigma is None:
                sigma = random.gammavariate(4, 1 / 4)
        else:
            k1 = 1
            k2 = 1
            b = 1
            sigma = 1
            # pick a random parameter to tweak
            match random.randrange(4):
                case 0:
                    k1 = random.gammavariate(4, 1 / 4)
                case 1:
                    k2 = random.gammavariate(4, 1 / 4)
                case 2:
                    b = random.gammavariate(4, 1 / 4)
                case 3:
                    sigma = random.gammavariate(4, 1 / 4)

        self.k1[node] = self.k1[node] * k1
        self.k2[node] = self.k2[node] * k2
        self.b[node] = self.b[node] * b
        self.sigma[node] = self.sigma[node] * sigma

    def negateBias(self, node=None):
        '''
        Negates the bias value. if node is not specified, negates a random
        node's bias
        '''
        if self.n < 1:
            return

        if node is None:
            node = self.randomNode()

        self.b[node] = self.b[node] * -1

    def mutate(self):
        '''
        applies a random mutation to the regulatory network.
        Hopefully these mutations are somewhat balances, so that on average
        the number of nodes and edges follow a random walk.
        '''

        # there are complemetary mutations that keep the number of nodes, and some
        #   without complements
        #   and edges on a random walk
        # addGene - deleteGene
        # split edge - deleteGene
        # flip edge
        # duplicate node - delete node
        # duplicate group - delete group
        # change index
        # change group index
        # create random edge - delete random edge (roughly even. delete advantage)
        # scale existing edge weight
        # negate weight
        # redirect existing edge (slightly uneven, sometimes deletes edges)
        # scale parameters
        # negate bias
        mutationGroup = random.randrange(13)
        choice = random.choice([True, False])
        match mutationGroup:
            case 0:
                # addGene - deleteGene
                if choice:
                    mutationIndex = 0
                else:
                    mutationIndex = 1
            case 1:
                # split edge - deleteGene
                if choice:
                    mutationIndex = 3
                else:
                    mutationIndex = 1
            case 2:
                # flip edge
                mutationIndex = 4
            case 3:
                # duplicate node - delete node
                if choice:
                    mutationIndex = 5
                else:
                    mutationIndex = 1
            case 4:
                # duplicate group - delete group
                if choice:
                    mutationIndex = 6
                else:
                    mutationIndex = 2
            case 5:
                # change index
                mutationIndex = 7
            case 6:
                # change group index
                mutationIndex = 8
            case 7:
                # create random edge - delete random edge (roughly even. delete advantage)
                if choice:
                    mutationIndex = 9
                else:
                    mutationIndex = 10
            case 8:
                # scale existing edge weight
                mutationIndex = 11
            case 9:
                # negate weight
                mutationIndex = 12
            case 10:
                # redirect existing edge (slightly uneven, sometimes deletes edges)
                mutationIndex = 13
            case 11:
                # scale parameters
                mutationIndex = 14
            case 12:
                # negate bias
                mutationIndex = 15

        # mutationIndex = 15
        match mutationIndex:
            case 0:
                print("addGene")
                self.addGene()
            case 1:
                print("removeGene")
                self.removeGene()
            case 2:
                print("removeGeneGroup")
                self.removeGeneGroup()
            case 3:
                print("splitEdge")
                self.splitEdge()
            case 4:
                print("flipEdge")
                self.flipEdge()
            case 5:
                print("duplicateNode")
                self.duplicateNode()
            case 6:
                print("duplicateNodeGroup")
                self.duplicateNodeGroup()
            case 7:
                print("changeNodeIndex")
                self.changeNodeIndex()
            case 8:
                print("changeNodeGroupIndex")
                self.changeNodeGroupIndex()
            case 9:
                print("addEdge")
                self.addEdge()
            case 10:
                print("removeEdge")
                self.removeEdge()
            case 11:
                print("scaleWeight")
                self.scaleWeight()
            case 12:
                print("negateWeight")
                self.negateWeight()
            case 13:
                print("redirectEdge")
                self.redirectEdge()
            case 14:
                print("scaleParameter")
                self.scaleParameter()
            case 15:
                print("negateBias")
                self.negateBias()


    def randomNode(self):
        '''
        returns the index of a random valid node.
        '''
        if self.n == 0:
            return None
            # random.randrange(
        ret = random.randrange(self.n)
        return ret

    def randomNodeRange(self, meanLength=3):
        '''
        returns a tuple representing a range of valid nodes
        (start, end) inclusive

        The average length of the range returned is meanLength void edge
        effects
        '''
        if self.n < 1:
            return None
        # choose random range, with a mean length
        r1 = self.randomNode()
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

        return nodeRange

    def randomEdge(self):
        '''
        returns a random valid edge, in the tuple format

            (parents, weight)

        where parents is a tuple

            (parent1, parent2)
        '''
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
        '''
        calculates the change in concentration for every gene product given
        the current concentration of those products, z.

        z is a list, length must be exactly self.n
        '''
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


class GNIstate:
    '''For managing state variable for input events and editor interface
    GeneNetworkInterfaceState'''
    MOUSEDOWNNOCOLLISION = 0
    MOUSEDOWNPAN = 1
    NEWEDGE = 2
    MOUSEDOWNMOVENODE = 3
    NONE = 4
    MOUSEHOVERNODE = 5
    MOUSEHOVEREDGE = 6
    MOUSEDOWNMOVED = 7
    NEWEDGEFINISHED = 8

    state = NONE
    mousePos = (0, 0)
    mouseDownPos = (0, 0)
    # begining node of new edge
    newEdgeStart = None
    newEdgeEnd = None
    newEdgeWeight = 1
    newEdge = None
    # node that's being moved
    movingNode = 0
    movingNodeOffset = (0, 0)
    hoveringNode = None

    showHelp = True

    helpString = [
     "Controls:",
     "h  - press to show/hide this help message",
     "leftclick  - add a gene",
     "The little circle that's changing size inside each gene is the",
     "   concentration of that gene's protein product, a morphogen, that",
     "   can interact with other genes to regulate their expression.",
     "leftclick and drag - create an interaction between two genes",
     "   Use the scrollwheel during and after dragging to adjust weight.",
     "   x  - during drag to cancel",
     "rightclick and drag    - move genes around",
     "middle click and drag  - pan the view",
     "scrollwheel   - zoom in and out",
     "while hovering over a gene, use the scrollwheel while holding down",
     "one of the following buttons to adjust the gene's parameters:",
     "   b   - bias, the basal/constituative expression input",
     "   m   - max, the maximum expression rate",
     "   d   - destroy, the kinetic degredation rate of the gene product",
     "i     - while hovering over a gene to toggle the persistent indicator",
     "display",
     "   Black horizontal bar is the basal expression input",
     "   Colored horizontal bars are the interaction inputs",
     "       which are proportional to the concentration of the connected",
     "       genes' products(morphogens)",
     "   The black curve illustrates the transfer function, which relates",
     "       the sum of a gene's regulator inputs to it's expression rate",
     "   Green vertical bar is the expression rate, how quickly the",
     "       concentration of the gene's product is increasing",
     "   Red vertical bar is the rate of degredation, which is",
     "       proportional to the current concentration.",
     "   The final rate of change of concentration is the sum of the red",
     "       and green bars."]


def geneNetworkEventHandler(event, geneNetwork, GNIstate):
    # basically a node editor for recurrent neural networks to model and
    # simulate gene regulatory networks
    #
    # left click and mouseup no collision no move to add gene (node)
    # middle click and drag to pan view
    # mousewheel and no collision to zoom view to/from mouse location
    # right click drag on node to move
    # left click drag on node to start edge creation (node to cursor)
    #   snap to node on hover collide (including starting node, make it curve)
    #   on mouseup, delete tmp edge unless snapped
    # right click drag on edge to add curve
    # left click drag up/down on edge to change weight proportional to value
    # or mousewheel while hover over edge to change weight
    # on hover over an edge, display the weight
    # press x while hovering over node to delete it
    # press x while hovering over edge to set it to zero (and stop rendering it
    # press spacebar to play/pause simulation (allow editing during sim)
    # press r to randomize/zero(+noise) concentrations
    # mousewheel while hovering over nodes adjusts parameters and
    # mousewheel while dragging creating a new edge to adjust the weight
    # after new edge is created, while mouse is hovering over the end node,
    #   scrolling the mousewheel adjusts the weight of the new edge
    # concentrations depending on what key is held down
    #   b is bias - offsets the constituative rate of expression before
    #       logistic curve is applied, so 0 in centered on .5
    #   m is max rate - k1 parameter, scales max expression rate
    #   r is removal rate constant - k2 parameter for kinetic equation
    #       of degredation
    #   d is diffusion rate - sigma parameter for gaussian diffusion kernel
    #   no key held down perturbs the concentration. simulation of that node
    #       is paused until mouse hovers off, so effectively the concentration
    #       is pinned at some value. (paused, or simply the concentration is
    #       reset every frame. or something)
    #       if the f key is pressed during this process, the node is toggled
    #           to stay fixed until f on hover is pressed again.
    # on hover over a node, display the concentration value
    # on hover while one of the above modifyer keys is held down, display
    #   the appropriate parameter value and name, like "k1=1"
    if event.type == pygame.KEYDOWN:
        GNIstate.mousePos = pygame.mouse.get_pos()
        collision = geneNetwork.checkNodeCollision(
            geneNetwork.screenToCords(GNIstate.mousePos))
        GNIstate.hoveringNode = collision

        match event.key:
            case pygame.K_h:
                # help
                GNIstate.showHelp = not GNIstate.showHelp

        match GNIstate.state:
            case GNIstate.NONE:
                match event.key:

                    case pygame.K_x:
                        # x it out
                        # delete a node if hovering
                        if GNIstate.hoveringNode is not None:
                            geneNetwork.net.removeGene(GNIstate.hoveringNode)
                            GNIstate.hoveringNode = None

                    case pygame.K_i:
                        # indicator toggle
                        # change state of indicators for node if hovering
                        if GNIstate.hoveringNode is not None:
                            geneNetwork.net.displayIndicators[
                                GNIstate.hoveringNode] = not geneNetwork.net.displayIndicators[
                                GNIstate.hoveringNode]

                    case pygame.K_z:
                        # Zap!
                        # choose a random mutation to apply to the network.
                        geneNetwork.net.mutate()

            case GNIstate.NEWEDGEFINISHED:
                match event.key:

                    case pygame.K_x:
                        # also delete the node, but be careful to clean up new
                        # edge state variables
                        if GNIstate.hoveringNode is not None:
                            geneNetwork.net.removeGene(GNIstate.hoveringNode)
                            GNIstate.hoveringNode = None
                            GNIstate.newEdgeStart = None
                            GNIstate.newEdgeEnd = None
                            GNIstate.newEdgeWeight = 1
                            GNIstate.newEdge = None
                            GNIstate.state = GNIstate.NONE

            case GNIstate.NEWEDGE:
                match event.key:

                    case pygame.K_x:
                        # cancel the edge creation
                        GNIstate.state = GNIstate.NONE
                        GNIstate.newEdgeEnd = None
                        GNIstate.newEdgeStart = None

    elif event.type == pygame.MOUSEBUTTONDOWN:
        GNIstate.mousePos = event.pos
        collision = geneNetwork.checkNodeCollision(
            geneNetwork.screenToCords(event.pos))
        GNIstate.hoveringNode = collision

        GNIstate.mouseDownPos = event.pos
        GNIstate.mousePos = event.pos
        # left mouse
        if event.button == 1:
            # if collision with node, start edge creation
            if GNIstate.hoveringNode is not None:
                # there is a collision, start making a new edge
                GNIstate.state = GNIstate.NEWEDGE
                # save colliding node index
                GNIstate.newEdgeStart = GNIstate.hoveringNode
                GNIstate.newEdgeEnd = GNIstate.hoveringNode

            else:
                # if no collisions, wait for mouse button up
                # (make sure no mouse move)
                # to add new node
                GNIstate.state = GNIstate.MOUSEDOWNNOCOLLISION
                # if collision with edge, start edge weight adjust
        elif event.button == 2:
            # middle mouse
            GNIstate.state = GNIstate.MOUSEDOWNPAN
        elif event.button == 3:
            # right mouse button
            if GNIstate.hoveringNode is not None:
                GNIstate.state = GNIstate.MOUSEDOWNMOVENODE
                GNIstate.movingNode = GNIstate.hoveringNode
                # store mouse offset from node center
                GNIstate.movingNodeOffset = v.sub(
                    geneNetwork.net.locs[GNIstate.hoveringNode],
                    geneNetwork.screenToCords(event.pos))

        elif event.button == 4:
            # scroll
            match GNIstate.state:
                case GNIstate.NONE:
                    if GNIstate.hoveringNode is not None:
                        # change some parameter if modifier is held, else, zoom
                        keys = pygame.key.get_pressed()
                        if keys[pygame.K_b]:
                            # adjust bias
                            geneNetwork.net.b[GNIstate.hoveringNode] += 0.1
                        elif keys[pygame.K_m]:
                            # adjust k1
                            geneNetwork.net.k1[GNIstate.hoveringNode] += 0.1
                        elif keys[pygame.K_r]:
                            # adjust k2
                            geneNetwork.net.k2[GNIstate.hoveringNode] += 0.1
                        elif keys[pygame.K_d]:
                            # adjust sigma
                            geneNetwork.net.sigma[GNIstate.hoveringNode] += 0.1
                        else:
                            # if no keys pressed, zoom.
                            geneNetwork.zoomTo(1.05,
                                               geneNetwork.screenToCords(
                                                event.pos))
                    else:
                        # if not hovering over node, zoom
                        geneNetwork.zoomTo(1.05,
                                           geneNetwork.screenToCords(
                                            event.pos))
                case GNIstate.NEWEDGE:
                    GNIstate.newEdgeWeight += 1

                case GNIstate.NEWEDGEFINISHED:
                    # in this case, mousewheel means adjust weight of edge
                    # change some parameter if modifier is held, else, weight
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_b]:
                        # adjust bias
                        geneNetwork.net.b[GNIstate.hoveringNode] += 0.1
                    elif keys[pygame.K_m]:
                        # adjust k1
                        geneNetwork.net.k1[GNIstate.hoveringNode] += 0.1
                    elif keys[pygame.K_r]:
                        # adjust k2
                        geneNetwork.net.k2[GNIstate.hoveringNode] += 0.1
                    elif keys[pygame.K_d]:
                        # adjust sigma
                        geneNetwork.net.sigma[GNIstate.hoveringNode] += 0.1
                    else:
                        # geneNetwork.net.w[GNIstate.newEdgeEnd][GNIstate.newEdgeStart] += 0.1
                        geneNetwork.net.setWeight(
                            GNIstate.newEdgeEnd,
                            GNIstate.newEdgeStart,
                            geneNetwork.net.getWeight(
                                GNIstate.newEdgeEnd,
                                GNIstate.newEdgeStart) + 0.1)

        elif event.button == 5:
            match GNIstate.state:
                case GNIstate.NONE:
                    if GNIstate.hoveringNode is not None:
                        # change some parameter if modifier is held, else, zoom
                        keys = pygame.key.get_pressed()
                        if keys[pygame.K_b]:
                            # adjust bias
                            geneNetwork.net.b[GNIstate.hoveringNode] -= 0.1
                        elif keys[pygame.K_m]:
                            # adjust k1, clamp positive
                            geneNetwork.net.k1[GNIstate.hoveringNode] = max(
                                geneNetwork.net.k1[GNIstate.hoveringNode] - 0.1,
                                0)
                        elif keys[pygame.K_r]:
                            # adjust k2, clamp positive
                            geneNetwork.net.k2[GNIstate.hoveringNode] = max(
                                geneNetwork.net.k2[GNIstate.hoveringNode] - 0.1,
                                0)
                        elif keys[pygame.K_d]:
                            # adjust sigma
                            geneNetwork.net.sigma[GNIstate.hoveringNode] = max(
                                geneNetwork.net.sigma[GNIstate.hoveringNode] -
                                0.1, 0)
                        else:
                            # if no keys pressed, zoom.
                            geneNetwork.zoomTo(1 / 1.05,
                                               geneNetwork.screenToCords(
                                                event.pos))
                    else:
                        # if not hovering over node, zoom
                        geneNetwork.zoomTo(1 / 1.05,
                                           geneNetwork.screenToCords(
                                            event.pos))
                case GNIstate.NEWEDGE:
                    GNIstate.newEdgeWeight -= 1

                case GNIstate.NEWEDGEFINISHED:
                    # in this case, mousewheel means adjust weight of edge
                    # change some parameter if modifier is held, else, weight
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_b]:
                        # adjust bias
                        geneNetwork.net.b[GNIstate.hoveringNode] -= 0.1
                    elif keys[pygame.K_m]:
                        # adjust k1, clamp positive
                        geneNetwork.net.k1[GNIstate.hoveringNode] = max(
                            geneNetwork.net.k1[GNIstate.hoveringNode] - 0.1,
                            0)
                    elif keys[pygame.K_r]:
                        # adjust k2, clamp positive
                        geneNetwork.net.k2[GNIstate.hoveringNode] = max(
                            geneNetwork.net.k2[GNIstate.hoveringNode] - 0.1,
                            0)
                    elif keys[pygame.K_d]:
                        # adjust sigma
                        geneNetwork.net.sigma[GNIstate.hoveringNode] = max(
                            geneNetwork.net.sigma[GNIstate.hoveringNode] -
                            0.1, 0)
                    else:
                        # geneNetwork.net.w[GNIstate.newEdgeEnd][GNIstate.newEdgeStart] -= 0.1
                        geneNetwork.net.setWeight(
                            GNIstate.newEdgeEnd,
                            GNIstate.newEdgeStart,
                            geneNetwork.net.getWeight(
                                GNIstate.newEdgeEnd,
                                GNIstate.newEdgeStart) - 0.1)

    elif event.type == pygame.MOUSEMOTION:
        GNIstate.mousePos = event.pos
        collision = geneNetwork.checkNodeCollision(
            geneNetwork.screenToCords(event.pos))
        GNIstate.hoveringNode = collision

        match GNIstate.state:
            case GNIstate.MOUSEDOWNNOCOLLISION:
                # cancel create node if mouse moved too far
                if v.mag(v.sub(event.pos, GNIstate.mouseDownPos)) > 5:
                    GNIstate.state = GNIstate.MOUSEDOWNMOVED
            case GNIstate.MOUSEDOWNPAN:
                geneNetwork.pan = v.sub(geneNetwork.pan,
                                        v.div(event.rel, geneNetwork.zoom))
            case GNIstate.NEWEDGE:
                # check for collision with other node
                # if collision, finalize edge for now
                # not over another node yet
                # if not, remove finsih node and set finish position to
                # mouse
                GNIstate.newEdgeEnd = GNIstate.hoveringNode

            case GNIstate.MOUSEDOWNMOVENODE:
                geneNetwork.net.locs[GNIstate.movingNode] = v.sum(
                    geneNetwork.screenToCords(event.pos),
                    GNIstate.movingNodeOffset)

            case GNIstate.NEWEDGEFINISHED:
                if GNIstate.hoveringNode is not None:
                    # see if we're still hovering over new node parent
                    if GNIstate.hoveringNode == GNIstate.newEdgeEnd:
                        # still hovering over new node, just wait
                        pass
                    else:
                        # stop newedgefinished
                        GNIstate.state = GNIstate.NONE
                        GNIstate.newEdgeEnd = None
                        GNIstate.newEdgeStart = None
                else:
                    # stop newedgefinished
                    GNIstate.state = GNIstate.NONE
                    GNIstate.newEdgeEnd = None
                    GNIstate.newEdgeStart = None

    elif event.type == pygame.MOUSEBUTTONUP:
        GNIstate.mousePos = event.pos
        collision = geneNetwork.checkNodeCollision(
            geneNetwork.screenToCords(event.pos))
        GNIstate.hoveringNode = collision

        # ignore scroll events or other mouse buttons
        if event.button == 1 or event.button == 2 or event.button == 3:
            match GNIstate.state:
                # check for collisions...
                case GNIstate.MOUSEDOWNPAN:
                    # do nothing
                    GNIstate.state = GNIstate.NONE

                case GNIstate.MOUSEDOWNNOCOLLISION:
                    GNIstate.state = GNIstate.NONE
                    # create a node
                    newNode = geneNetwork.net.addGene(
                        geneNetwork.screenToCords(event.pos))
                    # set hovering node
                    GNIstate.hoveringNode = newNode

                case GNIstate.MOUSEDOWNMOVED:
                    # do nothing
                    GNIstate.state = GNIstate.NONE

                case GNIstate.NEWEDGE:
                    if GNIstate.newEdgeEnd is not None:
                        # set weight for new edge
                        # geneNetwork.net.w[GNIstate.newEdgeEnd][GNIstate.newEdgeStart] = GNIstate.newEdgeWeight
                        geneNetwork.net.setWeight(
                            GNIstate.newEdgeEnd,
                            GNIstate.newEdgeStart,
                            GNIstate.newEdgeWeight)
                        # GNIstate.newEdgeEnd = None
                        # GNIstate.newEdgeStart = None
                        if GNIstate.newEdgeWeight == 0:
                            GNIstate.state = GNIstate.NONE
                        else:
                            GNIstate.state = GNIstate.NEWEDGEFINISHED
                        # reset new edge weight var
                        GNIstate.newEdgeWeight = 1

                    else:
                        # Never intersected finish node, clear the edge
                        GNIstate.state = GNIstate.NONE
                        GNIstate.newEdgeEnd = None
                        GNIstate.newEdgeStart = None

                case GNIstate.MOUSEDOWNMOVENODE:
                    GNIstate.state = GNIstate.NONE


class GeneNetwork:
    # contains all the functions needed for the event handler to interact
    # with a RegulatoryNetwork object and handles for the pygame loop to
    # update the simulation and render the information in a useable way
    #
    # single cell network.
    # concentrations
    # screen positions
    # renderer for nodes and edges
    #   pan and zoom transform included for every dimension
    #   render arrow or flat for positive/negative interaction, and color
    #   render thickness proportional to weight
    #   render edges with boundary from originating node
    #   render nodes as circles, randomized color
    #       with a red boarder if it's fixed
    #   render concentrations at each node as
    #       circle fill in alpha for concentration
    #       small circle inside with radius equal to concentration
    #       another ring inside with variable radius for expression Rate
    # renderer for a graph window, concentrations over time of all the nodes,
    #   color coded lines
    # maybe some bars for the other parameters?
    #   pre degradation, pre-sigmoid expression rate bar, stacked sub bars
    #       Horizontal, underneith a sigmoid curve?
    #       sub bar indicating basal contribution (b)
    #       sub bar for each input indicating contribution (wij*zj)
    #   Sigmoidal curve showing the relationship between sum of weights
    #   and resulting rate of expression rate.
    #       multiplied by k1 to show max expression rate
    #   pre-degredation, post-sigmoid bar indicating expression rate
    #       Vertical
    #       multiplied by k1
    #   degredation rate bar, starting from top of pre-degredation bar,
    #       adjacent to pre-deg, post-sigmoid (to visualize subtraction)
    #       Vertical
    #   total expression rate bar starting from origin, going to tip of
    #       degredation rate bar, equals final dz/dt for node
    # mouse collision check function for nodes and edges
    # add gene (node) function
    # add edges (non zero weight) function
    #   curvable edges for asthetics and self interaction feedback loops
    #       3 point curve of some kind
    # modify weights function
    # run simulation
    #   ability to fix some concentrations

    node_r = 20
    node_t = 1.5
    arrowGap_w = 2
    arrowTip_l = 7
    arrowTip_a = math.pi / 5
    arrowLoopback_a = math.pi * 0.8
    arrowLoopbackTip_a = math.pi / 3
    indicators_w = 3
    indicatorSigmoid_w = node_r
    indicators_dzdt_scale = node_r * 0.8
    indicators_z_scale = node_r * 0.1
    indicatorColor_b = (0, 0, 0)
    indicatorColor_k2 = (255, 0, 0)
    f = None

    def __init__(self, net, z):
        # self.net = RegulatoryNetwork()
        self.net = net
        self.z = z

        self.pan = [0, 0]
        self.zoom = 1

    def cordsToScreen(self, loc):
        # return vsub(vmul(loc, self.zoom), self.pan)
        return v.mul(v.sub(loc, self.pan), self.zoom)

    def screenToCords(self, loc):
        # return vdiv(vsum(loc, self.pan), self.zoom)
        return v.sum(v.div(loc, self.zoom), self.pan)

    def zoomTo(self, zoom, loc):
        panToMouse = v.sub(loc, self.pan)
        # self.pan = vsum(self.pan, vsub(panToMouse, vmul(panToMouse, zoom)))
        self.pan = v.sum(v.sub(v.mul(panToMouse, zoom), panToMouse), self.pan)
        self.zoom *= zoom

    def checkNodeCollision(self, testPoint):
        for (i, loc) in enumerate(self.net.locs):
            if v.mag(v.sub(testPoint, loc)) < self.node_r:
                # testPoint collided with a node
                return i
        # if no collision, return None
        return None

    def checkEdgeCollision(self, testPoint):
        pass

    def drawArrow(self, surface, startNode, endNode=None, endPos=(0, 0),
                  width=None,
                  weight=None,
                  color=None):
        # draw an arrow
        if color is None:
            color = (0, 0, 0)
        radius = 1
        angle = self.arrowTip_a

        if endNode is not None:
            if startNode != endNode:
                # draw between two given nodes
                unit = v.unit(v.sub(self.net.locs[endNode],
                                    self.net.locs[startNode]))
                end = v.sub(self.net.locs[endNode],
                            v.mul(unit, self.arrowGap_w + self.node_r))
            else:
                angle = self.arrowLoopbackTip_a

            # calculate width from weight
            # if weight is negative, draw flat arrowhead
            if self.net.getWeight(endNode, startNode) < 0:
                angle = math.pi / 2
            # calculate arrow thickness from weight
            radius = abs((self.net.f(
                self.net.getWeight(endNode, startNode) / 10) * 2 - 1) * 5)
        else:
            # draw to the given corrdinate from the given node
            unit = v.unit(v.sub(endPos, self.net.locs[startNode]))
            end = endPos

        # override radius if width or weight ar set
        if width is not None:
            # use given width
            radius = width
        if weight is not None:
            if weight < 0:
                angle = math.pi / 2
            else:
                angle = self.arrowTip_a
            radius = abs((self.net.f(
                weight / 10) * 2 - 1) * 5)

        if startNode != endNode:
            # draw a straight arrow line
            start = v.sum(self.net.locs[startNode],
                          v.mul(unit, self.arrowGap_w + self.node_r))
            perpUnit = v.perp(unit)
            pygame.draw.polygon(
                surface,
                color,
                [v.vint(self.cordsToScreen(
                    v.sum(start, v.mul(perpUnit, radius)))),
                 v.vint(self.cordsToScreen(
                     v.sum(start, v.mul(perpUnit, -radius)))),
                 v.vint(self.cordsToScreen(
                     v.sum(end, v.mul(perpUnit, -radius)))),
                 v.vint(self.cordsToScreen(
                     v.sum(end, v.mul(perpUnit, radius))))])

        else:
            # draw a curved arrow line
            # overwrite unit and end so the arrow heads draw in the right place
            # this section is a mess

            unit = v.rot(v.u(math.pi / 2), -self.arrowLoopback_a + math.pi)
            perpUnit = v.perp(unit)
            # this end point would be correct after I finish the arc calculator
            # end = vsum(vsum(vmul(unit, -(self.node_r + self.arrowGap_w)),
            #                 vmul(perpUnit, -radius)),
            #            self.locs[endNode])

            arcRadius = self.node_r * 0.5
            arcRect = pygame.Rect(0, 0,
                                  arcRadius * 2 * self.zoom,
                                  arcRadius * 2 * self.zoom)
            arcRect.center = self.cordsToScreen(
                v.sum(v.mul(v.u(-math.pi / 2 + math.pi / 8),
                            self.node_r * 1.2),
                      self.net.locs[endNode]))
            end = v.sum(self.screenToCords(arcRect.center),
                        v.mul(v.u(math.pi - self.arrowLoopback_a),
                              arcRadius - radius))
            pygame.draw.arc(surface, color,
                            arcRect,
                            -(math.pi - self.arrowLoopback_a),
                            math.pi,
                            max(int(radius * 2 * self.zoom), 1))

        # draw arrow heads
        aunit = v.rot(unit, math.pi - angle)
        perpaunit = v.perp(aunit)
        pygame.draw.polygon(
            surface,
            color,
            [v.vint(self.cordsToScreen(
                v.sum(end, v.mul(perpUnit, radius)))),
             v.vint(self.cordsToScreen(
                 v.sum(end, v.sum(
                    v.mul(aunit, self.arrowTip_l),
                    v.mul(perpUnit, radius))))),
             v.vint(self.cordsToScreen(
                 v.sum(end, v.sum(v.sum(
                    v.mul(aunit, self.arrowTip_l),
                    v.mul(perpaunit, radius * 2)),
                                v.mul(perpUnit, radius))))),
             v.vint(self.cordsToScreen(
                 v.sum(end, v.sum(v.mul(perpUnit, radius),
                                v.mul(perpaunit, radius * 2)))))])
        aunit = v.rot(unit, -(math.pi - angle))
        perpaunit = v.perp(aunit)
        pygame.draw.polygon(
            surface,
            color,
            [v.vint(self.cordsToScreen(
                v.sum(end, v.mul(perpUnit, -radius)))),
             v.vint(self.cordsToScreen(
                 v.sum(end, v.sum(
                    v.mul(aunit, self.arrowTip_l),
                    v.mul(perpUnit, -radius))))),
             v.vint(self.cordsToScreen(
                 v.sum(end, v.sum(v.sum(
                    v.mul(aunit, self.arrowTip_l),
                    v.mul(perpaunit, -radius * 2)),
                                v.mul(perpUnit, -radius))))),
             v.vint(self.cordsToScreen(
                 v.sum(end, v.sum(v.mul(perpUnit, -radius),
                                  v.mul(perpaunit, -radius * 2)))))])

    def drawIndicators(self, surface, i, indicators):
        # draw bars to indicate input contributions, constituative
        # expression rate, degredation rate, and total dz/dt
        # draw b bar
        # draw diffusion gaussian curve
        origin = v.sum(self.cordsToScreen(self.net.locs[i]),
                       (-self.indicatorSigmoid_w / 2 * self.zoom,
                        self.indicators_w * 2 * self.zoom))
        sum = self.net.b[i]
        length = self.net.b[i] * self.indicators_z_scale * self.zoom
        width = self.indicators_w * self.zoom
        if length < 0:
            # if negative, offset b bar down
            origin = v.sum(origin, (0, width))
        # end = vsum(origin, (0, -length))
        end = v.sum(origin, (length, 0))
        pygame.draw.polygon(surface,
                            self.indicatorColor_b,
                            [v.vint(origin),
                             v.vint(v.sum(origin, (0, -width))),
                             v.vint(v.sum(end, (0, -width))),
                             v.vint(end)])
        if length < 0:
            # re-offset up
            end = v.sum(end, (0, -width))

        for (color, length) in indicators:
            # tally up a sum of all the input lengths plus bias
            sum += length
            # draw all the positive lengths first
            if length > 0:
                # draw an indicator bar
                origin = end
                length = length * self.indicators_z_scale * self.zoom
                end = v.sum(origin, (length, 0))
                pygame.draw.polygon(surface,
                                    color,
                                    [v.vint(origin),
                                     v.vint(v.sum(origin, (0, -width))),
                                     v.vint(v.sum(end, (0, -width))),
                                     v.vint(end)])
        # offset the next bank on indicators
        end = v.sum(end, (0, -width))
        for (color, length) in indicators:
            # draw all the negative lengths now
            if length < 0:
                # draw an indicator bar
                origin = end
                length = length * self.indicators_z_scale * self.zoom
                end = v.sum(origin, (length, 0))
                pygame.draw.polygon(surface,
                                    color,
                                    [v.vint(origin),
                                     v.vint(v.sum(origin, (0, -width))),
                                     v.vint(v.sum(end, (0, -width))),
                                     v.vint(end)])
        sigx = end

        # draw post-sigmoid expression rate
        origin = v.sum(self.cordsToScreen(self.net.locs[i]),
                       (-self.indicators_w * 0 * self.zoom, 0))
        length = self.net.f(sum) * self.net.k1[i] * self.indicators_dzdt_scale * self.zoom
        end = v.sum(origin, (0, -length))
        sigy = v.sum(end, (width, 0))
        pygame.draw.polygon(surface,
                            pygame.Color("Green"),
                            [v.vint(origin),
                             v.vint(v.sum(origin, (width, 0))),
                             v.vint(v.sum(end, (width, 0))),
                             v.vint(end)])

        # draw degredation rate
        origin = v.sum(end, (width, 0))
        length = -self.net.k2[i] * self.z[i] * self.indicators_dzdt_scale * self.zoom
        end = v.sum(origin, (0, -length))
        pygame.draw.polygon(surface,
                            pygame.Color("Red"),
                            [v.vint(origin),
                             v.vint(v.sum(origin, (width, 0))),
                             v.vint(v.sum(end, (width, 0))),
                             v.vint(end)])

        # draw sigmoidal function lines
        pygame.draw.line(surface,
                         pygame.Color("Black"),
                         sigx,
                         (sigx[0], sigy[1]))
        pygame.draw.line(surface,
                         pygame.Color("Black"),
                         sigy,
                         (sigx[0], sigy[1]))
        # draw sigmoid function
        fi = []
        origin = v.sum(self.cordsToScreen(
                         self.net.locs[i]),
                       (-self.indicatorSigmoid_w / 2 * self.zoom, 0))
        for p in self.f:
            fi.append(v.sum(self.cordsToScreen(
                v.sum(
                    self.net.locs[i],
                    (p[0], -p[1] * self.net.k1[i]))),
                           (-self.indicatorSigmoid_w / 2 * self.zoom, 0)))
        pygame.draw.lines(surface,
                          pygame.Color("Black"),
                          False,
                          fi,
                          2)
        # defin sigmoid function samples at this scale
        self.g = []
        fmin = -self.indicatorSigmoid_w / 2 / self.indicators_z_scale
        fmax = self.indicatorSigmoid_w / 2 / self.indicators_z_scale
        fstep = 1 / self.indicators_z_scale / self.zoom
        fsteps = min(max(int((fmax - fmin) / fstep), 2), 30)
        fstep = (fmax - fmin) / fsteps
        for step in range(fsteps):
            x = fmin + fstep * step
            self.g.append((x * self.indicators_z_scale,
                           CellArray.gaussian(None, x, self.net.sigma[i]) *
                           self.indicators_dzdt_scale))
        # draw diffusion gaussian
        fi = []
        for p in self.g:
            fi.append(v.sum(self.cordsToScreen(
                v.sum(
                    self.net.locs[i],
                    (p[0], -p[1]))),
                           (-self.indicatorSigmoid_w / 2 * self.zoom, 0)))
        pygame.draw.lines(surface,
                          pygame.Color("Black"),
                          False,
                          fi,
                          2)
        # draw axis
        pygame.draw.line(surface,
                         pygame.Color("Black"),
                         v.sum(origin,
                               (-self.indicatorSigmoid_w / 2 * self.zoom,
                                0)),
                         v.sum(origin,
                               (self.indicatorSigmoid_w * self.zoom,
                                0)))

    def update(self, dt):
        # simulate the network using the built in single cell concentration
        # array z.
        self.net.step(self.z, dt)

    def render(self, surface, GNIstate, debug=False):

        # defin sigmoid function samples at this scale
        self.f = []
        fmin = -self.indicatorSigmoid_w / 2 / self.indicators_z_scale
        fmax = self.indicatorSigmoid_w / 2 / self.indicators_z_scale
        fstep = 1 / self.indicators_z_scale / self.zoom
        fsteps = min(max(int((fmax - fmin) / fstep), 2), 30)
        fstep = (fmax - fmin) / fsteps
        for i in range(fsteps):
            x = fmin + fstep * i
            self.f.append((x * self.indicators_z_scale,
                           self.net.f(x) * self.indicators_dzdt_scale))

        if debug:
            font = pygame.font.Font(None, 24)

        # draw nodes
        # TODO: don't call pygame.draw if the shape is far off surface, it
        # causes it to draw VERY slowely as it creates a HUGE area to pixelfill
        for i in range(self.net.n):
            pygame.draw.circle(surface,
                               self.net.colors[i],
                               v.vint(self.cordsToScreen(self.net.locs[i])),
                               int(self.node_r * self.zoom),
                               max(int(self.node_t * self.zoom), 1))
            # draw concentration indicator
            if self.z[i] == 0:
                radius = 0
            else:
                radius = int(1 / (1 + 1 / self.z[i]) *
                             self.node_r * 0.9 * self.zoom)
            pygame.draw.circle(surface,
                               self.net.colors[i],
                               v.vint(self.cordsToScreen(self.net.locs[i])),
                               radius)
            # draw index number
            if debug:
                text = font.render(i.__str__(),
                                   True,
                                   (0, 0, 0))
                textpos = text.get_rect()
                textpos.center = self.cordsToScreen(self.net.locs[i])
                surface.blit(text, textpos)

            indicators = []
            # also draw edges
            # TODO: don't call pygame.draw if the shape is far off surface, it
            # causes it to draw VERY slowely as it creates a HUGE area to
            # pixelfill
            for j in range(self.net.n):
                # for each edge into node i, draw if weight is not 0
                if self.net.getWeight(i, j) != 0:
                    # draw an arrow
                    self.drawArrow(surface, j, i)

                    color = pygame.Color(self.net.colors[j])
                    if j == i:
                        # if it's self, adjust the color so it's visible
                        color.hsva = (color.hsva[0],
                                      color.hsva[1],
                                      color.hsva[2] * 0.8)
                    # add length for indicator to draw
                    indicators.append((color,
                                       self.net.getWeight(i, j) *
                                       self.z[j]))

            # draw indicators for node if enabled or hovered
            if GNIstate.hoveringNode is not None:
                if GNIstate.hoveringNode == i:
                    self.drawIndicators(surface, i, indicators)
                else:
                    if self.net.displayIndicators[i]:
                        self.drawIndicators(surface, i, indicators)
            elif self.net.displayIndicators[i]:
                self.drawIndicators(surface, i, indicators)

        # draw new edge
        if GNIstate.state == GNIstate.NEWEDGE:
            # new edge color
            if GNIstate.newEdgeWeight == 0:
                color = (255, 0, 0)
                weight = 1
            else:
                color = (0, 255, 0)
                weight = GNIstate.newEdgeWeight

            if GNIstate.newEdgeEnd is not None:
                # over another node draw to other node
                self.drawArrow(surface,
                               GNIstate.newEdgeStart,
                               GNIstate.newEdgeEnd,
                               weight=weight,
                               color=color)
            else:
                # not over another node, draw to cursor
                self.drawArrow(surface,
                               GNIstate.newEdgeStart,
                               endPos=self.screenToCords(GNIstate.mousePos),
                               weight=weight,
                               color=color)

        if GNIstate.showHelp:
            font = pygame.font.Font(None, 24)
            nextPos = surface.get_rect()
            for helpLine in GNIstate.helpString:
                text = font.render(helpLine,
                                   True,
                                   (0, 0, 0))
                textpos = text.get_rect()
                textpos.topleft = nextPos.topleft
                nextPos.topleft = textpos.bottomleft
                surface.blit(text, textpos)


class CellArray:
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

    def __init__(self, size):

        # concentration array, 2 dimensions of space x # of morphogens
        self.z = [
            [[] for y in range(size[1])]
            for x in range(size[0])]

        self.geneNetwork = RegulatoryNetwork()

        self.geneNetwork.addConcentrationFunc = self.addGene
        self.geneNetwork.removeConcentrationFunc = self.removeGene

    def gaussian(self, x, sigma):
        if sigma == 0:
            if x == 0:
                return 1
            else:
                return 0
        else:
            return math.exp(-x**2 / (2 * sigma**2)) / \
                    math.sqrt(2 * math.pi * sigma**2)

    def step(self, dt):
        # copy the concentration array
        newz = []
        for (x, column) in enumerate(self.z):
            newz.append([])
            for (y, z) in enumerate(column):
                newz[x].append([])
                for zn in self.z[x][y]:
                    newz[x][y].append(zn)

        xmax = len(self.z)
        ymax = len(column)
        # TODO these series of kernels and for loops should be done as shaders
        # on a GPU, not like this
        for i in range(self.geneNetwork.n):
            # calculate kernel size for each morphogen, as each one
            # can have a unique diffusion constant (sigma, molecular
            # size, membrane diffusion rate, etc)
            kernelHalf = math.ceil(3 * self.geneNetwork.sigma[i])
            kernelSize = int(2 * kernelHalf + 1)
            kernelSum = 0
            kernelValues = []
            kernelOffsets = list(range(-kernelHalf, kernelHalf + 1))

            # calculate kernel parameters and normalize
            for x in kernelOffsets:
                kernelValues.append(self.gaussian(x,
                                                  self.geneNetwork.sigma[i]))
                kernelSum += kernelValues[-1]
            for (j, value) in enumerate(kernelValues):
                kernelValues[j] = value / kernelSum

            # diffusion kernel horizontal
            for (x, column) in enumerate(self.z):
                for (y, z) in enumerate(column):
                    # compute the diffusion kernel, one axis at a time
                    kernel = 0
                    for (j, kernelValue) in enumerate(kernelValues):
                        kernel += newz[
                            (x + kernelOffsets[j]) % xmax][y][i] \
                            * kernelValue
                    newz[x][y][i] = kernel
            # diffusion kernel vertical
            for (x, column) in enumerate(self.z):
                for (y, z) in enumerate(column):
                    # compute the diffusion kernel, one axis at a time
                    kernel = 0
                    for (j, kernelValue) in enumerate(kernelValues):
                        kernel += newz[x][
                            (y + kernelOffsets[j]) % ymax][i] \
                            * kernelValue
                    newz[x][y][i] = kernel
                    # self.z now contains the effect of the diffusion kernel
                    # in two dimensions

            # geneNetwork dt. For now, all morphogens are diffusive
            for (x, column) in enumerate(self.z):
                for (y, z) in enumerate(column):
                    # calculate_dt
                    dz = self.geneNetwork.calculate_dz(z)
                    for (i, dzi) in enumerate(dz):
                        newz[x][y][i] = max(newz[x][y][i] + dzi * dt, 0)

            for (x, column) in enumerate(newz):
                for (y, z) in enumerate(column):
                    for (i, newzi) in enumerate(z):
                        # sum together all changes to be made to z
                        # all integrations map to the producers coordinates, onto z
                        # integrate intracell morphogens, persistant
                        # integrate diffusion morphogens, persistant
                        # integrate intercell, persistant
                        self.z[x][y][i] = newzi
                        pass
            # self.z = newz

    # callback functions to manage self.z
    def addGene(self, newIndex):

        for column in self.z:
            for z in column:
                z.insert(newIndex, random.expovariate(1))

    def removeGene(self, index):

        for column in self.z:
            for z in column:
                z.pop(index)

    def update(self, dt):
        self.step(dt)

    def render(self, surface):
        # render the concentration values to an array of pixels

        for (x, column) in enumerate(self.z):
            for (y, z) in enumerate(column):
                # for testing I'll just render the first 3 concentrations
                # as rgb
                pygame.gfxdraw.pixel(surface, x, y, (min(z[0] * 255, 255),
                                                     min(z[1] * 255, 255),
                                                     min(z[2] * 255, 255)))
                # min(z[2] * 255, 255)))
