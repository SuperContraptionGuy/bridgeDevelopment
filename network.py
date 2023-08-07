import math


class Cell:
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
    '''

    def __init__(self,
                 n,
                 z=None,
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
        if z is None:
            self.z = [0 for x in range(n)]
        else:
            self.z = z
        if k1 is None:
            self.k1 = [1 for x in range(n)]
        else:
            self.k1 = k1
        if w is None:
            self.w = [[0 for x in range(n)] for y in range(n)]
        else:
            self.w = w
        if b is None:
            self.b = [0 for x in range(n)]
        else:
            self.b = b
        if k2 is None:
            self.k2 = [0 for x in range(n)]
        else:
            self.k2 = k2

    def modifyWeight(self, j, i, wij):
        # change weight of edge directed from node j to node i
        self.w[i][j] = wij

    def addGene(self, z, k1, w, w2, b, k2):
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
        self.z.append(z)
        self.k1.append(k1)
        self.b.append(b)
        self.k2.append(k2)
        for (i, w2i) in enumerate(w2):
            self.w[i].append(w2i)
        self.w.append(w)

    def f(self, x):
        # Transfer function, sigmoidal function
        return 1 / (1 + math.exp(-x))

    def calculate_dz(self):
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
        z = self.z
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

    def step(self, dt):
        # do the integration on the self stored concentration matrix.
        # only useful if the Network object isn't interacting with other
        # Network objects.

        dz = self.calculate_dz()

        # update the concentrations based on the expression rates
        for (i, dzi) in enumerate(dz):
            self.z[i] += dzi * dt


cell = Cell(3,
            z=[0.5, 0.5, 0.4],
            k1=[1, 1, 1],
            w=[[0, -1000, 0],
               [0, 0, -1000],
               [-1000, 0, 0]],
            b=[3, 3, 3],
            k2=[2, 2, 2])


for i in range(100000):
    cell.step(0.1)
    print('......')
    string = ''
    for morphogen in cell.z:
        string += '->|' + '='*int(morphogen * 500) + '\n'
    print(string)
    for i in range(500000):
        pass


class Cells:
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
    '''

    def __init__(self):
        # let's just make a grid for testing
        pass

    pass
