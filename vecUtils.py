import math
import pygame.color
import random


def sum(a, b):
    return (a[0] + b[0], a[1] + b[1])


def mul(a, b):
    return (a[0] * b, a[1] * b)


def div(a, b):
    return (a[0] / b, a[1] / b)


def sub(a, b):
    # return vsum(a, vmul(b, -1))
    return (a[0] - b[0], a[1] - b[1])


def mag(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1])


def unit(a):
    # return math.sqrt(math.pow(a[0], 2) + math.pow(a[1], 2))
    if a[0] == 0:
        if a[1] == 0:
            # if a is mag 0, return this default vector
            return (1, 0)
    return div(a, mag(a))


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def angle(a):
    """
    return the angle the given vector makes with the x axis
    """
    length = mag(a)
    if a[1] >= 0:
        return math.acos(min(a[0] / length, 1))
    else:
        return -math.acos(min(a[0] / length, 1))


def u(theta):
    """
    generate a unit vector at the specified angle
    """
    return (math.cos(theta), math.sin(theta))


def rot(a, theta):
    """
    rotate a vector by an angle
    """
    return mul(u(angle(a) + theta), mag(a))


def perp(a):
    '''
    get vector perpendicular to given, using right hand rule
    '''
    return (-a[1], a[0])


def vint(vec):
    '''
    return integer representation of vector
    '''
    return (int(vec[0]), int(vec[1]))


def randomHue(color):
    newColor = pygame.Color(color)
    hsva = newColor.hsva
    hsva = (random.randint(0, 360), hsva[1], hsva[2], hsva[3])
    newColor.hsva = hsva
    return newColor
