# controls: Left mouse button and drag to create nodes connected to the
# previous node, or connected to the node clicked on. Nodes can be merged.
# Right mouse button and drag to move existing nodes. Nodes can be merged.
# d while hovering over a node to delete it.
import os
import json
import math
import random
import pygame
import pygame.gfxdraw
import bridge

import Box2D

import cProfile
import pstats


def load_png(name):
    """ Load image and return image object"""
    fullname = os.path.join(data_dir, name)
    try:
        image = pygame.image.load(fullname)
        if image.get_alpha() is None:
            image = image.convert()
        else:
            image = image.convert_alpha()
    except FileNotFoundError:
        print(f"Cannot load image: {fullname}")
        raise SystemExit
    return image, image.get_rect()


class DebugCircle(pygame.sprite.Sprite):

    def __init__(self, pos, radius=10, color=(255, 0, 0)):
        pygame.sprite.Sprite.__init__(self)
        self.pos = pos
        self.radius = radius
        self.color = color
        self.updateImage()

    def update(self):
        self.rect.center = self.pos

    def updateImage(self):
        self.image = pygame.Surface((self.radius*2, self.radius*2))
        KEY = (255, 255, 255)
        self.image.fill(KEY)
        self.image.set_colorkey(KEY)
        pygame.draw.circle(self.image, self.color,
                           (self.radius, self.radius), self.radius)
        self.image.set_alpha(150)
        self.rect = self.image.get_rect()
        self.update()


class DebugLine(pygame.sprite.Sprite):

    def __init__(self, pos1, pos2, thickness, color):
        pygame.sprite.Sprite.__init__(self)
        self.thickness = thickness
        self.radius = thickness / 2
        self.pos1 = pos1
        self.pos2 = pos2
        self.color = color
        self.updatePosition()

    def updatePosition(self):
        p1 = self.pos1
        p2 = self.pos2
        width = abs(p1[0] - p2[0])
        height = abs(p1[1] - p2[1])

        # calculate the sprite local corrdinates and sprite location (p)
        # depending on parent relative locations

        p = p1
        p1l = (self.radius, self.radius)
        if p1[0] >= p2[0]:
            p1l = (width + self.radius, p1l[1])
            p = (p2[0], p[1])
        if p1[1] >= p2[1]:
            p1l = (p1l[0], height + self.radius)
            p = (p[0], p2[1])

        p2l = (self.radius, self.radius)
        if p2[0] >= p1[0]:
            p2l = (width + self.radius, p2l[1])
        if p2[1] >= p1[1]:
            p2l = (p2l[0], height + self.radius)

        self.image = pygame.Surface((self.radius * 2 + width,
                                     self.radius * 2 + height))
        KEY = (255, 255, 255)
        self.image.fill(KEY)
        self.image.set_colorkey(KEY)
        pygame.draw.line(self.image, self.color,
                         p1l, p2l, width=self.thickness)
        self.rect = self.image.get_rect()
        self.rect.move_ip(p)


class CarGameObject(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.orig_image, self.rect = load_png("car.png")
        self.b2body = None
        self.angle = 0
        self.rotateImage()
        self.rect.center = (-200, -200)

    def update(self):
        if self.b2body is not None:
            self.rect.center = box2dToPygame(self.b2body.position)
            self.angle = self.b2body.angle / 2 / math.pi * 360
            self.rotateImage()

    def rotateImage(self):
        self.image = pygame.transform.rotate(self.orig_image, self.angle)
        self.rect = self.image.get_rect(center=self.rect.center)


class NodeGameObject(pygame.sprite.Sprite):
    """ A game object that handels rendering and user interaction with the
    node objects"""

    def __init__(self, node=bridge.Node()):
        pygame.sprite.Sprite.__init__(self)
        self.node = node
        # load an image from disk. Alternatively, render using graphic
        # functions
        self.image, self.rect = load_png("node.png")
        self.updatePosition()
        self.moving = False
        self.simulate = False

    def update(self):
        # do stuff like update position based on mouse inputs
        if self.simulate:
            self.rect.center = box2dToPygame(self.node.b2body.position)
        elif self.moving:
            self.updatePosition()

    def startMoving(self):
        self.moving = True

    def stopMoving(self):
        self.moving = False

    def updatePosition(self):
        self.rect.center = self.node.location

    def getSerializableObject(self):
        return {"location": self.node.location,
                "type": self.node.type}

    def loadSerializableObject(serializableObject):
        location = serializableObject["location"]
        type = serializableObject["type"]
        return NodeGameObject(bridge.Node(location, type))


class EdgeGameObject(pygame.sprite.Sprite):
    """game object that handles rendering and user interaction of edge objects
    """

    def __init__(self, edge):
        pygame.sprite.Sprite.__init__(self)
        self.edge = edge
        # print(edge.parents[0].location, edge.parents[1].location)
        self.moving = False
        self.radius = self.edge.thickness / 2
        self.updatePosition()
        self.tension = 0
        self.simulate = False

    def update(self, invdt=0):
        # Sprite update
        if self.simulate and not self.edge.broken:

            # update the tension value from joint reaction forces, if there
            # are two joints still attached
            if len(self.edge.joints) >= 2:
                # difference in joint forces (force differential referenced[0],
                # that's parent[0])
                appliedForce = vsub(self.edge.joints[1]
                                    .GetReactionForce(invdt),
                                    self.edge.joints[0]
                                    .GetReactionForce(invdt))
                # find direction of edge (edge vector)
                edgeVector = vunit(vsub(self.edge.parents[1].b2body.position,
                                        self.edge.parents[0].b2body.position))
                # find differential force in direction of edge
                self.tension = vdot(appliedForce, edgeVector)
            elif self.edge.type == "cable" and len(self.edge.joints) >= 1:
                # for cable types
                self.tension = 2 * vmag(self.edge.joints[0]
                                        .GetReactionForce(invdt))
            else:
                # other/unknown type
                self.tension = 0

            tensions.append(self.tension)

        if self.moving or self.simulate and not self.edge.broken:
            self.updatePosition()

    def startMoving(self):
        self.moving = True

    def stopMoving(self):
        self.moving = False

    def updatePosition(self):
        if not self.edge.broken:
            if simulate:
                p1 = box2dToPygame(self.edge.parents[0].b2body.position)
                p2 = box2dToPygame(self.edge.parents[1].b2body.position)
            else:
                p1 = self.edge.parents[0].location
                p2 = self.edge.parents[1].location
            width = abs(p1[0] - p2[0])
            height = abs(p1[1] - p2[1])

            # calculate the sprite local corrdinates and sprite location (p)
            # depending on parent relative locations

            p = p1
            p1l = (self.radius, self.radius)
            if p1[0] >= p2[0]:
                p1l = (width + self.radius, p1l[1])
                p = (p2[0], p[1])
            if p1[1] >= p2[1]:
                p1l = (p1l[0], height + self.radius)
                p = (p[0], p2[1])

            p2l = (self.radius, self.radius)
            if p2[0] >= p1[0]:
                p2l = (width + self.radius, p2l[1])
            if p2[1] >= p1[1]:
                p2l = (p2l[0], height + self.radius)

            self.image = pygame.Surface((self.radius * 2 + width,
                                         self.radius * 2 + height))
            KEY = (255, 255, 255)
            self.image.fill(KEY)
            self.image.set_colorkey(KEY)
            pygame.draw.line(self.image, self.edge.color,
                             p1l, p2l, width=self.edge.thickness)
            self.rect = self.image.get_rect()
            self.rect.move_ip(p)
        else:
            # empty image for broken edge
            self.image = pygame.Surface((0, 0))

    def getSerializableObject(self, nodes):
        parent1Obj = self.edge.parents[0]
        parent2Obj = self.edge.parents[1]
        p1done = False
        p2done = False
        for (i, node) in enumerate(nodes):
            if node.node == parent1Obj and not p1done:
                parent1 = i
                p1done = True
            if node.node == parent2Obj and not p2done:
                parent2 = i
                p2done = True
            if p1done and p2done:
                break
        else:
            # no parents, for some reason
            if not p1done:
                parent1 = None
            if not p2done:
                parent2 = None

        return {"parents": [parent1, parent2],
                "type": self.edge.type}

    def loadSerializableObject(serializableObject, nodes):
        parent1 = serializableObject["parents"][0]
        parent2 = serializableObject["parents"][1]

        # Make sure there was a parent
        if parent1 is not None:
            parent1Obj = nodes[parent1].node
        else:
            parent1Obj = bridge.Node()
        if parent2 is not None:
            parent2Obj = nodes[parent2].node
        else:
            parent2Obj = bridge.Node()

        type = serializableObject["type"]
        return EdgeGameObject(bridge.Edge((parent1Obj,
                                           parent2Obj), type))


def serializeBridge():
    pass


def vsum(a, b):
    return (a[0] + b[0], a[1] + b[1])


def vmul(a, b):
    return (a[0] * b, a[1] * b)


def vdiv(a, b):
    return (a[0] / b, a[1] / b)


def vsub(a, b):
    # return vsum(a, vmul(b, -1))
    return (a[0] - b[0], a[1] - b[1])


def vmag(a):
    # return math.sqrt(math.pow(a[0], 2) + math.pow(a[1], 2))
    return math.sqrt(a[0] * a[0] + a[1] * a[1])


def vunit(a):
    return vdiv(a, vmag(a))


def vdot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def vangle(a):
    """
    return the angle the given vector makes with the x axis
    """
    length = vmag(a)
    if a[1] >= 0:
        return math.acos(min(a[0] / length, 1))
    else:
        return -math.acos(min(a[0] / length, 1))


def u(theta):
    """
    generate a unit vector at the specified angle
    """
    return (math.cos(theta), math.sin(theta))


def vrot(a, theta):
    """
    rotate a vector by an angle
    """
    return vmul(u(vangle(a) + theta), vmag(a))


def pygameToBox2d(position):
    """
    switch coordinate systems from pygame pixels to Box2D meters
    """
    position = (position[0], screen.get_height() - position[1])
    position = vdiv(position, PPM)
    return position


def box2dToPygame(position):
    """
    switch coordinate systems from Box2D meters to pygame pixels
    """
    position = vmul(position, PPM)
    position = (position[0], SCREEN_HEIGHT - position[1])
    return position


def randomHue(color):
    newColor = pygame.Color(color)
    hsva = newColor.hsva
    hsva = (random.randint(0, 360), hsva[1], hsva[2], hsva[3])
    newColor.hsva = hsva
    return newColor


def calculateNewPosition(goalPos):
    """
    Calculates the new position of a nodes[-1] respecting the edge length
    constraints of it's connected edges.
    """
    originalPosition = nodes[-1].node.location
    nodes[-1].node.location = goalPos
    # calculate new position based on cusor position and max
    # edge lengths
    criticalPoints = []
    debugCriticalPoints.clear()
    reachable = True

    # tolerance added to any distance comparison maximum to ensure floating
    # point errors are not a cause for distance check failures, which causes
    # unexpected results
    tolerance = 0.00000000001

    # reorder the parents tuple so that node[-1] is first
    # this check to to make sure r1 corrisponds to nodes[-1]
    # that is, r1 corrisponds to the node trying to reach the goal position
    # this loop honestly doesn't need to run everytime calculateNewPosition()
    # is called, only if the node[-1] is changed, or edges are added
    # cannot be run in the for edge loop after this, since not all nodes will
    # be reconfigured before they are used.
    for edge in connectedEdges:
        if edge.edge.parents.index(nodes[-1].node) != 0:
            edge.edge.parents = (edge.edge.parents[1], edge.edge.parents[0])

    for edge in connectedEdges:

        # see which edges can reach cursor, calculate closest point for those
        # that can't reach and add that point to the criticalPoints list if
        # it's within range of all other edges
        # one circle solutions

        r1 = edge.edge.parents[0].location
        r2 = edge.edge.parents[1].location

        r = edge.edge.maxLength

        # add debug circle showing reachable area for each connected edge
        # debugCriticalPoints.append(DebugCircle(r2, r, randomHue(debugRadii_c)))
        if debug:
            debugCriticalPoints.append(DebugCircle(r2, r, debugRadii_c))

        rVecInRange = False
        if math.dist(r2, r1) > r + tolerance:
            # found one edge that didn't reach? Notify end logic to use a
            # critical point from the list we're about to populate
            reachable = False
            # calculate nearest point to cursor
            rVec = vsum(r2, vmul(vunit(vsub(r1, r2)), r))
            if debug:
                debugCriticalPoints.append(DebugCircle(rVec, 3, (255, 0, 0)))

            # check range for all other Connected edges
            # only search if rVecInRange is initially True, otherwise it's
            # not been defined above.
            rVecInRange = True

        edges2 = connectedEdges[:]
        edges2.remove(edge)
        # we must iterate all edge2's every time, not just in last if statement
        for edge2 in edges2:
            # Now, check if there is a second edge that can't reach the
            # goal position. Also check for other solution types, like
            # the intersection of two circles. Both of these kinds of
            # solutions are 'criticalPoints' and the critica point closest
            # to the goal position will be the solution, IF
            # at least one of the edges can't reach the goal
            # position directly

            # second parent is not nodes[-1](not the node I'm trying to locate)
            r2_2 = edge2.edge.parents[1].location
            r_2 = edge2.edge.maxLength

            # distance from goal point to second test point
            # if math.dist(r1_2, r2_2) > r_2:

            # Check range of one (two really) circle solution
            # That is, on the boundary of the first circle, as close
            # as possible to the goal, is that point still within
            # range of the second circle?
            # first, check if it's been defined
            if rVecInRange:
                # now check if it's in range
                if math.dist(rVec, r2_2) > r_2 + tolerance:
                    # the possible critical point is not within range for
                    # second edge. Otherwise, it's a valid point
                    rVecInRange = False
            # Now, every other connected edge has been checked against the
            # rVec solution for this particular edge, an edge that couldn't
            # reach the goal directly. rVec represents the point closest to
            # the goal position but still within range of edge's maxLength,
            # that is a point in the boundary of the circle centered at edge's
            # other parent with radius maxLength. if rVecInRange is true,
            # then this point is within range of every other connectedEdge.
            # otherwise it's not within the maxLength of at leas one other edge

            # add to criticalPoints if in range of others (one circle solution)
            # once per connected edge. So, this should compile a list
            # of points on the boundary of circles around nodes with connected
            # edges that cannot reach the goal position.

            # we must now find some candidate criticalPoints in addition or
            # instead of rVec.

            # in this case, at least edge is out
            # of range of the goal position. Now, find if the two intersection
            # points between edge's range and edge2's ranges
            # are within range of all other edges
            # that is, find the intersection points of every pair of
            # circles.
            # If an intersection is also within range of all the other
            # edges, it's a critical point

            C_1r = r2
            C_2r = r2_2
            d_v = vsub(C_2r, C_1r)
            d = vmag(d_v)
            r1_c = r
            r2_c = r_2
            # cases without solutions (degenerates):
            # d > r1 + r2       - too far apart, no intersection
            # d = 0             - circles on top of each other, none or
            #                     infinite solutions
            # d < | r1 - r2 |   - one circle inside the other with no overlap
            # make sure none of these cases occur before calculating P, or
            # else math error will ensue
            if not (d > r1_c + r2_c or d == 0 or d < abs(r1_c - r2_c)):

                # the max(min(...)) is to make sure we're within the domain of
                # acos and asin. sometimes we're out of bounds by a floating
                # rounding error

                # psudo code for quadrant dependant expression
                # phi = D.x > 0 ?
                #   asin(D.y / d) : D.y > 0 ?
                #       acos(D.x / d) : -acos(D.x / d),
                if d_v[0] > 0:
                    phi = math.asin(max(min(d_v[1] / d, 1), -1))
                elif d_v[1] > 0:
                    phi = math.acos(max(min(d_v[0] / d, 1), -1))
                else:
                    phi = -math.acos(max(min(d_v[0] / d, 1), -1))

                # theta = math.acos(max(min((math.pow(r1_c, 2) +
                #                            math.pow(d, 2) -
                #                            math.pow(r2_c, 2))
                #                           /
                #                           (2 * r1_c * d), 1), -1))
                # rewrote without math.pow
                theta = math.acos(max(min((r1_c * r1_c +
                                           d * d -
                                           r2_c * r2_c)
                                          /
                                          (2 * r1_c * d), 1), -1))

                # A solution to the two circle intersection problem. This point
                # lies on the perimeter of both circles, positioned at
                # C_r1 and C_r2, with radii of r1_c and r2_c which corrispond
                # to the position of two noded connected to nodes[-1] with
                # radii of the maxLength of the respective connecting edges
                P = vsum(C_1r, vmul(u(phi + theta), r1_c))
                if debug:
                    # red dots are all the discoverd intersection solutions
                    debugCriticalPoints.append(DebugCircle(P, 3, (255, 0, 0)))

                # check if P is in range of all other connectedEdges
                edges3 = edges2[:]
                edges3.remove(edge2)
                # PinRange = True
                for edge3 in edges3:
                    # second parent is not nodes[-1]
                    # r2_3 = edge3.edge.parents[1].location
                    # r_3 = edge2.edge.maxLength

                    if math.dist(edge3.edge.parents[1].location, P) > edge3.edge.maxLength + tolerance:
                        # not in range of something
                        # PinRange = False
                        break
                else:
                    # else: runs if there was no break, ie, P was never out of
                    # range of any of the edges
                    # if PinRange:
                    # subset of solutions also colored white later on will the
                    # the allowed solutions, those that are within range of
                    # every connected edge
                    # this list of critical points will contain the final
                    # calculated position if the goal cannot be reached
                    # directly
                    criticalPoints.append(P)
            else:
                # some edge case occured. There was no solution for the
                # intersection of these two circles, so no criticalPoints
                # probably due to some problem with the node positions, or
                # two nodes connected by an edge were positioned right on
                # top of each other.
                # maybe we detect and handle these in the future?
                pass

        # checked all the other edges to see if they are in range for rVec
        # if none of the distance measurement failed, and rVec was defined
        # initially, then it's a valid critical point.
        if rVecInRange:
            criticalPoints.append(rVec)

    if debug:
        for point in criticalPoints:
            # add all found critical points to debug display
            # white circles show valid solutions, all within range of every
            # connected edge. The closest of these to the goal will the the
            # final solution, if the goal is not directly reachable
            debugCriticalPoints.append(DebugCircle(point, 5, (254, 254, 254)))

    # it's not possible to set the location to the goal position. We must
    # choose the closest criticalPoint.
    # check to make sure goal wasn't directly reachable
    if not reachable:
        # check if there are any criticalPoints to use, that some valid
        # solution was found
        if len(criticalPoints) > 0:

            # now find the point in criticalPoints that's closest to the goal
            # closestPoint = criticalPoints[0]
            closestDist = -1
            for point in criticalPoints:
                # find distance to goalPos
                dist = math.dist(point, goalPos)
                # compare with the closest so far
                if closestDist < 0 or dist < closestDist:
                    # make new closestPoint and set closestDist if this is
                    # the first point, or if it's closer than the previous
                    # closest point
                    closestDist = dist
                    closestPoint = point

            nodes[-1].node.location = closestPoint
            # return False, meaning did not reach the goal position
            return False
        else:
            # so, goal isn't reachable, and math failed to find a criticalPoint
            # so that means the constraints were too complicated, probably
            # off by just a rounding error, or the nodes were positioned
            # beforehand in a way that broke the constraints.
            # the node shouldn't be moved at all.
            nodes[-1].node.location = originalPosition
            return False

    # was able to reach goal position directly? return true
    # That is, every edge is less than or equal to it's maxLength when we
    # move
    # the movable node  is moved to the goal position
    # node[-1] has alread been moved to the goal poisition at the top
    # of this function, so nothing more to do.
    return True


def resetBridge():
    # create some anchor points
    anchor1pos = (screen.get_width() / 2 + 75 * 3,
                  screen.get_height() / 2)
    anchor2pos = (screen.get_width() / 2 - 75 * 3,
                  screen.get_height() / 2)
    nodes.append(NodeGameObject(bridge.Node(anchor1pos)))
    nodes[-1].node.type = "fixed"
    nodes.append(NodeGameObject(bridge.Node(anchor2pos)))
    nodes[-1].node.type = "fixed"


main_dir = os.path.split(os.path.abspath(__file__))[0]
data_dir = os.path.join(main_dir, "data")

# initialize
pygame.init()
screen = pygame.display.set_mode((1280, 720))
pygame.display.set_caption("PolyBridge")
clock = pygame.time.Clock()
running = True
# time step in milliseconds
dt = 0

debugs = []
debugCriticalPoints = []
debugDeleteLines = []
debug = False
debugCursor = DebugCircle((0, 0), 5)
debugs.append(debugCursor)

colorObj = pygame.Color(0, 0, 0)
colorObj.hsva = (216, 64, 100)
debugRadii_c = pygame.Color(colorObj)
colorObj.hsva = (0, 100, 100)
debugPoints_c = pygame.Color(colorObj)

background = pygame.Surface(screen.get_size())
background = background.convert()
background.fill((42, 234, 148))
anchor1pos = (screen.get_width() / 2 + 75 * 3,
              screen.get_height() / 2)
anchor2pos = (screen.get_width() / 2 - 75 * 3,
              screen.get_height() / 2)
pygame.draw.rect(background, (255, 255, 255),
                 (anchor1pos,
                  (screen.get_width() - anchor1pos[0],
                   screen.get_height() / 2)))
pygame.draw.rect(background, (255, 255, 255),
                 ((0, anchor2pos[1]),
                  (anchor2pos[0], screen.get_height() / 2)))

# add text to background
font = pygame.font.Font(None, 36)
text = font.render("Bio-Polybridge!", 1, (234, 148, 42))
textpos = text.get_rect()
textpos.centerx = background.get_rect().centerx
background.blit(text, textpos)


nodes = []
edges = []
car = CarGameObject()
collidingNode = None
connectedEdges = []


edgeMode = "road"
simulate = False

# Physics engine stuff, Box2D

# --- constants ---
# Box2D deals with meters, but we want to display pixels,
# so define a conversion factor:
PPM = 20.0  # pixels per meter
TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = screen.get_width(), screen.get_height()
# inverse timestep in inverse seconds
invdt = TARGET_FPS

colors = {
    Box2D.b2_staticBody: (255, 255, 255, 100),
    Box2D.b2_dynamicBody: (127, 127, 127, 100),
}

# --- pybox2d world setup ---
# Create the world
world = Box2D.b2World(gravity=(0, -10), doSleep=True)

# list of bodies for simulation
box2dBodies = []

resetBridge()

# update sprite arrays
carSprite = pygame.sprite.RenderPlain(car)
nodeSprites = pygame.sprite.RenderPlain(nodes)
edgeSprites = pygame.sprite.RenderPlain(edges)
debugSprites = pygame.sprite.RenderPlain(debugs)

pr = cProfile.Profile()
pr.enable()

tensions = []
steps = 0
# game loop runs once per 1/60th of a second.
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_F3:
                debug = not debug

            if event.key == pygame.K_d and not simulate:
                mousepos = pygame.mouse.get_pos()
                # check for nodes colliding with cursor to delete
                # ([:] for copy of list since I'm modifying it while iterating)
                for node in nodes:
                    if node.rect.collidepoint(mousepos) and node.node.type == "movable":
                        # find edges that include this node and remove them
                        for edge in edges[:]:
                            if node.node in edge.edge.parents:
                                edges.remove(edge)
                        nodes.remove(node)

                        # stop after the first hit
                        break

                if debug:
                    debugDeleteLines.clear()
                for edge in edges:
                    # if edge.rect.collidepoint(mousepos):

                    # the cursor has intersected the sprite, but is it
                    # close enough to the line to really count?
                    a = vsub(mousepos, edge.edge.parents[0].location)
                    amag = vmag(a)
                    b = vsub(edge.edge.parents[1].location,
                             edge.edge.parents[0].location)
                    bunit = vunit(b)
                    adotbunit = vdot(a, bunit)
                    d = math.sqrt(max(amag * amag - adotbunit * adotbunit, 0))

                    if adotbunit > 0 and adotbunit < vmag(b):
                        # it's within the range of the edge (it's right next to
                        # it)

                        if debug:
                            # adds red lines representing the vector from the
                            # line to be deleted and the cursor for each line
                            # actually tested. if an in range hit occurs, not
                            # all lines will be tested.
                            bbase = vsum(vmul(bunit, adotbunit),
                                         edge.edge.parents[0].location)
                            # debugEdgeDist.pos1 = bbase
                            # debugEdgeDist.pos2 = mousepos
                            # debugEdgeDist.updatePosition()
                            debugDeleteLines.append(
                                    DebugLine(bbase, mousepos, 3, (255, 0, 0)))

                        # check range
                        if d < 10:
                            edges.remove(edge)
                            # just remove one edge
                            break

                nodeSprites = pygame.sprite.RenderPlain(nodes)
                edgeSprites = pygame.sprite.RenderPlain(edges)

            if event.key == pygame.K_1:
                edgeMode = "road"
            if event.key == pygame.K_2:
                edgeMode = "support"
            if event.key == pygame.K_3:
                edgeMode = "cable"

            if event.key == pygame.K_s and not simulate:
                # save the bridge
                jsonBridge = {"nodes": [],
                              "edges": []}

                for node in nodes:
                    jsonBridge["nodes"].append(
                            node.getSerializableObject())

                for edge in edges:
                    jsonBridge["edges"].append(
                            edge.getSerializableObject(nodes))

                with open('savedBridge', 'w', encoding="utf-8") as f:
                    f.write(json.dumps(jsonBridge))

            if event.key == pygame.K_l and not simulate:
                # load a bridge from file
                with open('savedBridge', 'r', encoding="utf-8") as f:
                    jsonString = f.read()

                jsonBridge = json.loads(jsonString)

                nodes.clear()
                edges.clear()

                for node in jsonBridge["nodes"]:
                    nodes.append(NodeGameObject.loadSerializableObject(node))

                for edge in jsonBridge["edges"]:
                    edges.append(
                            EdgeGameObject.loadSerializableObject(edge, nodes))

                nodeSprites = pygame.sprite.RenderPlain(nodes)
                edgeSprites = pygame.sprite.RenderPlain(edges)
                collidingNode = None

            if event.key == pygame.K_r and not simulate:
                nodes.clear()
                edges.clear()
                collidingNode = None
                connectedEdges = []
                resetBridge()
                nodeSprites = pygame.sprite.RenderPlain(nodes)
                edgeSprites = pygame.sprite.RenderPlain(edges)

            if event.key == pygame.K_SPACE:
                # clear the last simulation
                if simulate:
                    simulate = False

                    for node in nodes:
                        node.simulate = False
                        node.node.b2body = None
                        node.updatePosition()

                    for edge in edges:
                        edge.simulate = False
                        edge.edge.joints = []
                        edge.edge.broken = False
                        edge.edge.b2bodies.clear()
                        edge.updatePosition()

                    for body in box2dBodies:
                        world.DestroyBody(body)

                    car.b2body = None
                    box2dBodies.clear()

                else:
                    simulate = True
                    steps = 0

                    # rebuild the static ground
                    anchor1pos = (screen.get_width() / 2 + 75 * 3,
                                  screen.get_height() / 2 - 2.5)
                    anchor2pos = (screen.get_width() / 2 - 75 * 3,
                                  screen.get_height() / 2 - 2.5)
                    size = vdiv((
                            (screen.get_width() - anchor1pos[0]) / 2,
                            (screen.get_height() - anchor1pos[1]) / 2), PPM)

                    center = pygameToBox2d((
                        (screen.get_width() -
                            anchor1pos[0]) / 2 + anchor1pos[0],
                        (screen.get_height() -
                            anchor1pos[1]) / 2 + anchor1pos[1]))

                    ground_body_right = world.CreateStaticBody(
                            position=center,
                            shapes=Box2D.b2PolygonShape(box=size))
                    ground_body_right.fixtures[0].filterData.groupIndex = -1
                    ground_body_right.fixtures[0].filterData.categoryBits = 0x0002
                    ground_body_right.fixtures[0].filterData.maskBits = 0x0002

                    size = vdiv((
                            (anchor2pos[0]) / 2,
                            (screen.get_height() - anchor2pos[1]) / 2), PPM)

                    center = pygameToBox2d((
                        (anchor2pos[0]) / 2,
                        (screen.get_height() - anchor2pos[1]) / 2 + anchor2pos[1]))

                    leftShape = Box2D.b2PolygonShape(
                            box=(size[0], size[1],
                                 (center[0], center[1]), 0))

                    # And a static body to hold the ground shape
                    ground_body_left = world.CreateStaticBody(
                            position=center,
                            shapes=Box2D.b2PolygonShape(box=size))
                    ground_body_left.fixtures[0].filterData.groupIndex = -1
                    ground_body_left.fixtures[0].filterData.categoryBits = 0x0002
                    ground_body_left.fixtures[0].filterData.maskBits = 0x0002

                    box2dBodies.append(ground_body_left)
                    box2dBodies.append(ground_body_right)

                    # make the car
                    carwheelbase = 2.5
                    carheight = 1.5
                    carwheelradius = 0.5
                    carpos = vsum(pygameToBox2d(
                                (0, screen.get_height() / 2)),
                                              (carwheelbase,
                                               carheight / 2 + carwheelradius))
                    wheeloffset = vsum(carpos, (carwheelbase / 2,
                                                -carheight / 2))
                    wheel2offset = vsum(carpos,
                                        (-carwheelbase / 2,
                                         -carheight / 2))
                    carbody = world.CreateDynamicBody(
                            position=carpos,
                            shapes=[Box2D.b2PolygonShape(
                                box=(carwheelbase / 2, carheight / 2))],
                            shapeFixture=Box2D.b2FixtureDef(
                                density=1,
                                friction=0.3,
                                categoryBits=0x0002,
                                maskBits=0x0002))
                    wheel1 = world.CreateDynamicBody(
                            position=wheeloffset,
                            shapes=[Box2D.b2CircleShape(
                                radius=carwheelradius)],
                            shapeFixture=Box2D.b2FixtureDef(
                                density=1,
                                friction=1,
                                categoryBits=0x0002,
                                maskBits=0x0002))
                    wheel2 = world.CreateDynamicBody(
                            position=wheel2offset,
                            shapes=[Box2D.b2CircleShape(
                                radius=carwheelradius)],
                            shapeFixture=Box2D.b2FixtureDef(
                                density=1,
                                friction=1,
                                categoryBits=0x0002,
                                maskBits=0x0002))

                    world.CreateRevoluteJoint(
                            bodyA=carbody,
                            bodyB=wheel1,
                            anchor=wheeloffset,
                            motorSpeed=-10,
                            maxMotorTorque=50,
                            enableMotor=True)
                    world.CreateRevoluteJoint(
                            bodyA=carbody,
                            bodyB=wheel2,
                            anchor=wheel2offset,
                            motorSpeed=-10,
                            maxMotorTorque=50,
                            enableMotor=True)

                    car.b2body = carbody
                    box2dBodies.append(carbody)
                    box2dBodies.append(wheel1)
                    box2dBodies.append(wheel2)

                    # construct the physics simulation
                    for node in nodes:
                        # generate a body with circle shape fixed to it for
                        # each node
                        # change of coordinates
                        node.simulate = True
                        position = pygameToBox2d(node.node.location)

                        # body type depends on movable/fixed type of node
                        match node.node.type:

                            case "movable":
                                node.node.b2body = world.CreateDynamicBody(
                                        position=position)

                            case "fixed":
                                node.node.b2body = world.CreateStaticBody(
                                        position=position)

                        # node.node.b2body.CreateCircleFixture(
                        #         radius=0.5, friction=0.3)

                        # add the body to the list for simulation
                        box2dBodies.append(node.node.b2body)

                    for edge in edges:
                        edge.simulate = True
                        pos1 = pygameToBox2d(edge.edge.parents[0].location)
                        pos2 = pygameToBox2d(edge.edge.parents[1].location)
                        vec = vsub(pos2, pos1)
                        length = vmag(vec)
                        width = 5 / PPM
                        # find center point between parents
                        center = vsum(pos1, vdiv(vec, 2))

                        if edge.edge.type != "cable":
                            # check for bounds of acos function, l != 0
                            if length > 0:
                                # find angle
                                # if vec[1] >= 0:
                                #     angle = math.acos(min(vec[0] / length, 1))
                                # else:
                                #     angle = -math.acos(min(vec[0] / length, 1))
                                angle = vangle(vec)

                                edgeBody = world.CreateDynamicBody(
                                        position=center, angle=angle)
                            else:
                                # edge of zero length...
                                edgeBody = world.CreateDynamicBody(
                                        position=center, angle=0)

                            # now add a shape
                            edgeBody.CreatePolygonFixture(
                                    box=(length / 2, width / 2),
                                    density=1,
                                    friction=0.3)

                            # collisionfilter to keep edges from colliding with
                            # each other
                            edgeBody.fixtures[0].filterData.groupIndex = -1

                            # add collisions with car
                            if edge.edge.type == "road":
                                edgeBody.fixtures[0].filterData.categoryBits = 0x0002
                                edgeBody.fixtures[0].filterData.maskBits = 0x0002

                            # now create some joints to the node parents.
                            joint1 = world.CreateRevoluteJoint(
                                    bodyA=edge.edge.parents[0].b2body,
                                    bodyB=edgeBody,
                                    anchor=pos1)

                            joint2 = world.CreateRevoluteJoint(
                                    bodyA=edge.edge.parents[1].b2body,
                                    bodyB=edgeBody,
                                    anchor=pos2)

                            edge.edge.joints = [joint1, joint2]

                            edge.edge.b2bodies.append(edgeBody)
                            box2dBodies.append(edgeBody)

                        else:
                            # cable rope joint
                            # ropeJoint = world.CreateRopeJoint(
                            ropeJoint = world.CreateRopeJoint(
                                    bodyA=edge.edge.parents[0].b2body,
                                    bodyB=edge.edge.parents[1].b2body,
                                    anchorA=pygameToBox2d(
                                        edge.edge.parents[0].location),
                                    anchorB=pygameToBox2d(
                                        edge.edge.parents[1].location),
                                    collideConnected=True,
                                    maxLength=math.dist(
                                        edge.edge.parents[0].location,
                                        edge.edge.parents[1].location) / PPM)
                            edge.edge.joints.append(ropeJoint)

        if event.type == pygame.MOUSEBUTTONDOWN and not simulate:

            # on left mouse button: create new node, connected to last node
            # or to the node just clicked on.
            # on right mouse button: move the node just clicked on

            if event.button == 1:
                # left mouse button -> create new and move

                newNode = NodeGameObject(bridge.Node(event.pos))
                newNode.startMoving()
                nodes.append(newNode)
                nodeSprites = pygame.sprite.RenderPlain(nodes)

                # check if there are 2 nodes to connect with an edge
                if len(nodes) > 1:

                    # pr.enable()
                    # check if the curror was clicking on an existing node
                    connectingNode = nodes[-2].node

                    # check for collisions, not including the
                    # moving one(last one)
                    for node in nodes[0:-2]:
                        if node.rect.collidepoint(event.pos):
                            # if so, make edge between the clicked
                            # on node and new
                            connectingNode = node.node
                            # stop at first hit
                            break

                    # make the edge
                    newEdge = EdgeGameObject(
                        bridge.Edge(
                            (connectingNode,
                             nodes[-1].node), edgeMode))
                    newEdge.startMoving()
                    edges.append(newEdge)

                    edgeSprites = pygame.sprite.RenderPlain(edges)

                    # add to connectedEdges list
                    connectedEdges = []
                    connectedEdges.append(newEdge)
                    calculateNewPosition(event.pos)

                    # pr.disable()

            if event.button == 3:
                # right mouse button -> move existing if it's not fixed
                for node in nodes[:]:
                    if node.rect.collidepoint(event.pos) and node.node.type == "movable":
                        # pr.enable()
                        # move node to end of list, and begin moving
                        nodes.remove(node)
                        nodes.append(node)
                        node.startMoving()

                        # assign each connected edge to moving, and add it to
                        # connectedEdges list
                        connectedEdges = []
                        for edge in edges:
                            if node.node in edge.edge.parents:
                                edge.startMoving()
                                connectedEdges.append(edge)

                        calculateNewPosition(event.pos)
                        # pr.disable()
                        break

        if event.type == pygame.MOUSEBUTTONUP and not simulate:
            if len(nodes) > 0:
                nodeSprites.update()
                nodes[-1].stopMoving()
                edgeSprites.update()
                for edge in edges:
                    edge.stopMoving()

            if collidingNode is not None:
                # if there is a collision to resolve, connect all the edges
                # currently connected to new node to the collidingNode, then
                # delete new node, moving collidingNode to front.
                # nodes.remove(collidingNode)

                # Also, try to resolve duplicate edges, giving priority to
                # the most recent edges.

                """
                if collidingNode.node in edges[-1].edge.parents:
                    # degenerate case, two parents the same
                    # remove new node and remove new edge
                    nodes.pop()
                    edges.pop()
                else:
                    edges[-1].edge.parents = (edges[-1].edge.parents[0],
                                              collidingNode.node)
                """
                uniqueNodes = []
                # reverse order to give precidence to recent edges for
                # duplicate resolution
                edgesReversed = edges[:]
                edgesReversed.reverse()
                for edge in edgesReversed:
                    if nodes[-1].node in edge.edge.parents:
                        if collidingNode.node in edge.edge.parents:
                            # this is a degenerate case, the two merging nodes
                            # have a connecting edge. Remove this edge.
                            edges.remove(edge)
                            continue
                        else:
                            # re-assign the edge's parents
                            # Also keep track of the other connected Node, to
                            # check for other edges that reference the same
                            # node (other as in not nodes[-1] or collidingNode)
                            index = edge.edge.parents.index(nodes[-1].node)
                            if index == 0:
                                edge.edge.parents = (collidingNode.node,
                                                     edge.edge.parents[1])
                                uniqueNode = edge.edge.parents[1]
                            else:
                                edge.edge.parents = (edge.edge.parents[0],
                                                     collidingNode.node)
                                uniqueNode = edge.edge.parents[0]

                    # keep track of unique parent nodes to solve for duplicate
                    # edges
                    elif collidingNode.node in edge.edge.parents:
                        index = edge.edge.parents.index(collidingNode.node)
                        if index == 0:
                            uniqueNode = edge.edge.parents[1]
                        else:
                            uniqueNode = edge.edge.parents[0]

                    else:
                        # neither collidingNode or nodes[-1] are parents of
                        # this edge, do nothing more, skip the remaining
                        # section
                        continue

                    # here, either collidingNode or nodes[-1] are the parent,
                    # but not both, and not neither.
                    # uniqueNode has been defined.
                    # Now check for duplicate nodes
                    if uniqueNode in uniqueNodes:
                        # this node has been see already, and since the
                        # iteration is in reverse order, we prioritize
                        # the first sighting, therfore, delete this
                        # edge since it would otherwise contain the
                        # same parents as a previous, newer edge.
                        edges.remove(edge)
                    else:
                        # hasn't been see yet, so add it to the list
                        uniqueNodes.append(uniqueNode)

                # remove new node, and move collidingNode to the end
                nodes.remove(collidingNode)
                nodes.pop()
                nodes.append(collidingNode)

                nodeSprites = pygame.sprite.RenderPlain(nodes)
                edgeSprites = pygame.sprite.RenderPlain(edges)
                collidingNode = None

        if event.type == pygame.MOUSEMOTION and not simulate:
            debugCursor.pos = event.pos
            # check for collisions of mouse with other nodes, default none
            collidingNode = None
            if len(nodes) > 0:
                if nodes[-1].moving:
                    # pr.enable()
                    # move the node, check for max edge length for all
                    # connected edges
                    # set location of node to update parents of edges to cursor
                    # check for collisions,
                    # not including the moving one(last one)
                    reachedOtherNode = False
                    for node in nodes[0:-1]:
                        if node.rect.collidepoint(event.pos):
                            # set location of moving node to collision node
                            # (don't forget to remove the duplication later
                            # and resolve the edges)
                            # set goal position
                            if calculateNewPosition(node.node.location):
                                # nodes[-1].node.location = node.node.location
                                collidingNode = node
                            reachedOtherNode = True
                            # actually, I want to stop on the first hit, and
                            # just put the node as close as possible to it, to
                            # make designing easier
                            break

                    # if it can't reach the other node (calculateNewPosition
                    # evaluated false) then calculateNewPosition again against
                    # mouse position as the goal
                    if not reachedOtherNode:
                        calculateNewPosition(event.pos)

                    # pr.disable()

    keys = pygame.key.get_pressed()
    if keys[pygame.K_ESCAPE]:
        running = False

    mouse = pygame.mouse.get_pressed()
    if mouse[0]:
        # print("Mouse is down")
        pass

    # update sprites/game objects
    carSprite.update()
    nodeSprites.update()
    edgeSprites.update(invdt)
    if debug:
        debugCriticalPointsSprites = pygame.sprite.RenderPlain(
                debugCriticalPoints)
        debugDeleteLinesSprites = pygame.sprite.RenderPlain(debugDeleteLines)
        debugSprites.update()
        debugCriticalPointsSprites.update()
        debugDeleteLinesSprites.update()

    # let the simulation settle before breaking stuff
    if steps > 60 * 2 and simulate:
        for edge in edges:
            # check for breakage condition
            if edge.tension > edge.edge.tensileStrength or edge.tension < -edge.edge.compressiveStrength:
                if not edge.edge.broken:
                    edge.edge.broken = True
                    edge.tension = 0
                    edge.updatePosition()
                    # breakage condition, could do a fancy break beam in half
                    # thing. I'll just break the joint for now
                    # or just remove it from the simulation
                    if edge.edge.type != "cable" and len(edge.edge.b2bodies) > 0:
                        body = edge.edge.b2bodies.pop()
                        world.DestroyBody(body)
                        box2dBodies.remove(body)
                    elif len(edge.edge.joints) > 0:
                        # cable type, destroy the joint
                        world.DestroyJoint(edge.edge.joints.pop())
                    # world.DestroyJoint(edge.edge.joints.pop())

    # draw to the screen
    # clear the screen
    screen.blit(background, (0, 0))

    # Draw the physics world from Box2D's perspective
    if debug:
        # for body in box2dBodies:
        for body in world.bodies:
            # The body gives us the position and angle of its shapes
            for fixture in body.fixtures:
                # The fixture holds information like density and friction,
                # and also the shape.
                shape = fixture.shape
                match shape.type:

                    # render polygons for generic
                    case Box2D.b2Shape.e_polygon:
                        vertices = [(body.transform * v) * PPM
                                    for v in shape.vertices]

                        vertices = [(int(v[0]), int(SCREEN_HEIGHT - v[1]))
                                    for v in vertices]

                        # pygame.gfxdraw.aapolygon(screen,
                        #                          vertices,
                        #                          colors[body.type])
                        pygame.draw.polygon(screen,
                                            colors[body.type],
                                            vertices)

                    case Box2D.b2Shape.e_circle:
                        center = body.transform * shape.pos * PPM
                        center = (int(center[0]),
                                  int(SCREEN_HEIGHT - center[1]))
                        radius = int(shape.radius * PPM)

                        # pygame.gfxdraw.aacircle(screen,
                        #                         center[0],
                        #                         center[1],
                        #                         radius,
                        #                         colors[body.type])
                        pygame.draw.circle(screen,
                                           colors[body.type],
                                           center,
                                           radius)

    # render using pygame sprites
    carSprite.draw(screen)
    edgeSprites.draw(screen)
    if not simulate:
        nodeSprites.draw(screen)

    if debug:
        debugSprites.draw(screen)
        debugCriticalPointsSprites.draw(screen)
        debugDeleteLinesSprites.draw(screen)

    # draw the tension/compression forces
    # if simulate and steps > 60 * 2:
    if simulate:
        for edge in edges:
            tension = edge.tension
            edge1pos = box2dToPygame(edge.edge.parents[0].b2body.position)
            edge2pos = box2dToPygame(edge.edge.parents[1].b2body.position)
            edgeVec = vsub(edge2pos,
                           edge1pos)
            unitVec = vunit(edgeVec)
            length = tension * vmag(edgeVec) / 2
            arrowWidth = 2
            arrowAngle = 30 * 2 * math.pi / 360

            # set arrow colors. Grey if within settling period
            if steps > 60 * 2:
                arrowColorTension = (255, 0, 0)
                arrowColorCompression = (0, 0, 255)
            else:
                arrowColorTension = (200, 200, 200)
                arrowColorCompression = arrowColorTension

            if tension > 0:
                loc = vdiv(vsum(edge2pos,
                                edge1pos),
                           2)
                length /= edge.edge.tensileStrength

                head1 = vsum(vmul(unitVec, length), loc)
                head2 = vsum(vmul(unitVec, -length), loc)

                # Draw two outward facing arrows
                pygame.draw.line(screen, arrowColorTension,
                                 head2, head1,
                                 width=arrowWidth)

                pygame.draw.line(screen, arrowColorTension,
                                 head1,
                                 vsum(head1,
                                      vrot(vmul(unitVec, -length / 4), arrowAngle)),
                                 width=arrowWidth)
                pygame.draw.line(screen, arrowColorTension,
                                 head1,
                                 vsum(head1,
                                      vrot(vmul(unitVec, -length / 4), -arrowAngle)),
                                 width=arrowWidth)

                pygame.draw.line(screen, arrowColorTension,
                                 head2,
                                 vsum(head2,
                                      vrot(vmul(unitVec, length / 4), arrowAngle)),
                                 width=arrowWidth)
                pygame.draw.line(screen, arrowColorTension,
                                 head2,
                                 vsum(head2,
                                      vrot(vmul(unitVec, length / 4), -arrowAngle)),
                                 width=arrowWidth)
            elif tension < 0:
                length /= -edge.edge.compressiveStrength
                head1 = vsum(vmul(unitVec, length), edge1pos)
                head2 = vsum(vmul(unitVec, -length), edge2pos)
                # Draw two arrows facing inward
                pygame.draw.line(screen, arrowColorCompression, edge1pos,
                                 head1,
                                 width=arrowWidth)
                pygame.draw.line(screen, arrowColorCompression, edge2pos,
                                 head2,
                                 width=arrowWidth)

                pygame.draw.line(screen, arrowColorCompression,
                                 head1,
                                 vsum(head1,
                                      vrot(vmul(unitVec, -length / 4), arrowAngle)),
                                 width=arrowWidth)
                pygame.draw.line(screen, arrowColorCompression,
                                 head1,
                                 vsum(head1,
                                      vrot(vmul(unitVec, -length / 4), -arrowAngle)),
                                 width=arrowWidth)

                pygame.draw.line(screen, arrowColorCompression,
                                 head2,
                                 vsum(head2,
                                      vrot(vmul(unitVec, length / 4), arrowAngle)),
                                 width=arrowWidth)
                pygame.draw.line(screen, arrowColorCompression,
                                 head2,
                                 vsum(head2,
                                      vrot(vmul(unitVec, length / 4), -arrowAngle)),
                                 width=arrowWidth)

    # Make Box2D simulate the physics of our world for one step.
    # Instruct the world to perform a single step of simulation. It is
    # generally best to keep the time step and iterations fixed.
    # See the manual (Section "Simulating the World") for further discussion
    # on these parameters and their implications.
    world.Step(TIME_STEP, 30, 30)

    # Flip buffers
    pygame.display.flip()

    # clock.tick() returns seconds last call.
    dt = clock.tick(TARGET_FPS) / 1000
    # invdt = 1 / dt
    steps += 1

pr.disable()
# pr.create_stats()
pr.dump_stats("profile")
pstats.Stats("profile").sort_stats(
        pstats.SortKey.CUMULATIVE, pstats.SortKey.CALLS).print_stats()
pstats.Stats("profile").sort_stats(
        pstats.SortKey.CUMULATIVE, pstats.SortKey.CALLS).print_callers()
pstats.Stats("profile").sort_stats(
        pstats.SortKey.TIME, pstats.SortKey.CALLS).print_callers()
pygame.quit()
