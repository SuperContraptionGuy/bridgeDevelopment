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

    def __init__(self):
        '''
        var  description                size
        z   concentrations matrix @t    n
        k1  max expression rate         n
        w   directed weights matrix     n x n
        b   bias vector                 n
        k2  degradation rate            n
        '''
        # initalize internal neural network parameters
        self.n = 0

        self.z = []
        self.k1 = []
        self.k2 = []
        self.b = []

        self.edgeList = {}

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

        self.n += 1
        self.z.insert(newIndex, random.expovariate(1))
        self.displayIndicators.insert(newIndex, False)

        if copy is not None:
            # copy parameters from copy, but not edges
            self.k1.insert(newIndex, self.k1[copy])
            self.b.insert(newIndex, self.b[copy])
            self.k2.insert(newIndex, self.k2[copy])
            self.locs.insert(newIndex, self.locs[copy])
            self.colors.insert(newIndex, pygame.Color(self.colors[copy]))

        else:
            # populate with default values
            self.k1.insert(newIndex, random.expovariate(1))
            self.b.insert(newIndex, random.gauss(0, 1))
            self.k2.insert(newIndex, random.expovariate(1))
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

    def removeGene(self, i=None):
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

        self.k1.pop(i)
        self.b.pop(i)
        self.k2.pop(i)
        self.z.pop(i)
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
    # redirect existing edge
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
