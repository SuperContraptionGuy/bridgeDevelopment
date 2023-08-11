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
        if k1 is None:
            self.k1 = [1 for x in range(n)]
        else:
            self.k1 = k1
        if w is None:
            self.w = [[0 for x in range(n)] for y in range(n)]
        else:
            self.w = w
        if b is None:
            self.b = [1 for x in range(n)]
        else:
            self.b = b
        if k2 is None:
            self.k2 = [1 for x in range(n)]
        else:
            self.k2 = k2

    def modifyWeight(self, j, i, wij):
        # change weight of edge directed from node j to node i
        self.w[i][j] = wij
        # some possible mutations using this function:
        # • randomize proportional to current value. This modifies the strength
        # of intereaction one morphogen has on another only if they already
        # interact
        # • set weight to zero, breaking the interaction entierly
        # • setting small weights to zero, breaking weak iteractions
        # • set weight to some small number besides zero if it was already
        # zero, making a new interaction

    def addGene(self,
                k1=None,
                w=None,
                w2=None,
                b=None,
                k2=None):
        '''
        adds a node to the neural network (a gene/morphogen)
        all parameters are numbers, except w, which is a vector of length n
        (including the newly added node) which represents how the new node
        is affected by all other concentrations, and w2, which is a vector of
        length n-1 (not including the new node) which represents how every
        other node is affected by the new one. The last value in w represents
        the weight the new substance has on itself.
        '''
        self.n += 1
        if k1 is not None:
            self.k1.append(k1)
        else:
            self.k1.append(1)
        if b is not None:
            self.b.append(b)
        else:
            self.b.append(1)
        if k2 is not None:
            self.k2.append(k2)
        else:
            self.k2.append(1)

        if w is not None:
            # first add an entry on the inputs to every other node
            for (i, w2i) in enumerate(w2):
                self.w[i].append(w2i)
            # then add an entire row of inputs for the new node
            self.w.append(w)
        else:
            # fill with zeros, no connections between the old nodes and the
            # new nodes
            for i in range(self.n-1):
                self.w[i].append(0)
            self.w.append([0 for i in range(self.n)])

    def removeGene(self, i):
        '''Remove gene i'''
        if self.n > 0:
            self.n -= 1

            self.k1.pop(i)
            self.b.pop(i)
            self.k2.pop(i)

            # remove the input from gene i from all other nodes
            for j in self.w:
                j.pop(i)
                # does this work? TODO
            # remove entire row on inputs for i
            self.w.pop(i)

    def f(self, x):
        '''Sigmoidal function'''
        # make sure to avoid Overflow errors
        if -x > math.floor(math.log(sys.float_info.max)):
            return 0

        # Transfer function, sigmoidal function
        return 1 / (1 + math.exp(-x))

    def calculate_dz(self, z):
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
        dz = [0 for x in range(self.n)]
        # for each output node i, calculate rate of change of concentration
        for (i, zi) in enumerate(z):
            # bias vector
            gi = self.b[i]
            # for each input node j, sum the effect it has on output i
            for (j, zj) in enumerate(z):
                # maybe this if will make it a little bit faster
                if self.w[i][j] != 0:
                    gi += self.w[i][j] * zj

            dz[i] = self.k1[i] * self.f(gi) - self.k2[i] * zi

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
