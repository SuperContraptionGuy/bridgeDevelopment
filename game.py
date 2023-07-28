# controls: Left mouse button and drag to create nodes connected to the
# previous node, or connected to the node clicked on. Nodes can be merged.
# Right mouse button and drag to move existing nodes. Nodes can be merged.
# d while hovering over a node to delete it.
import os
import math
import random
import pygame
import bridge

import cProfile
import pstats

main_dir = os.path.split(os.path.abspath(__file__))[0]
data_dir = os.path.join(main_dir, "data")

# initialize
pygame.init()
screen = pygame.display.set_mode((1280, 720))
pygame.display.set_caption("PolyBridge")
clock = pygame.time.Clock()
running = True
dt = 0


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

    def update(self):
        # do stuff like update position based on mouse inputs
        if self.moving:
            self.updatePosition()

    def startMoving(self):
        self.moving = True

    def stopMoving(self):
        self.moving = False

    def updatePosition(self):
        self.rect.center = self.node.location


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

    def update(self):
        # Sprite update
        if self.moving:
            self.updatePosition()

    def startMoving(self):
        self.moving = True

    def stopMoving(self):
        self.moving = False

    def updatePosition(self):
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


# generate a unit vector at the specified angle
def u(theta):
    return (math.cos(theta), math.sin(theta))


debugs = []
debugCriticalPoints = []
debugDeleteLines = []
debug = False
debugCursor = DebugCircle((0, 0), 5)
debugs.append(debugCursor)


def randomHue(color):
    newColor = pygame.Color(color)
    hsva = newColor.hsva
    hsva = (random.randint(0, 360), hsva[1], hsva[2], hsva[3])
    newColor.hsva = hsva
    return newColor


colorObj = pygame.Color(0, 0, 0)
colorObj.hsva = (216, 64, 100)
debugRadii_c = pygame.Color(colorObj)
colorObj.hsva = (0, 100, 100)
debugPoints_c = pygame.Color(colorObj)


def calculateNewPosition(goalPos):
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


background = pygame.Surface(screen.get_size())
background = background.convert()
background.fill((42, 234, 148))


# add text to background
font = pygame.font.Font(None, 36)
text = font.render("Bio-Polybridge!", 1, (234, 148, 42))
textpos = text.get_rect()
textpos.centerx = background.get_rect().centerx
background.blit(text, textpos)


nodes = []
edges = []
collidingNode = None
connectedEdges = []


# create some anchor points
nodes.append(NodeGameObject(bridge.Node(
    (screen.get_width() / 4, screen.get_height() / 2))))
nodes[-1].node.type = "fixed"
nodes.append(NodeGameObject(bridge.Node(
    (screen.get_width() * 3 / 4, screen.get_height() / 2))))
nodes[-1].node.type = "fixed"

# update sprite arrays
nodeSprites = pygame.sprite.RenderPlain(nodes)
edgeSprites = pygame.sprite.RenderPlain(edges)
debugSprites = pygame.sprite.RenderPlain(debugs)

edgeMode = "road"

pr = cProfile.Profile()
pr.enable()

# game loop runs once per 1/60th of a second.
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_F3:
                debug = not debug

            if event.key == pygame.K_d:
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
                    d = math.sqrt(amag * amag - adotbunit * adotbunit)

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
                        if d < 20:
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

        if event.type == pygame.MOUSEBUTTONDOWN:

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

        if event.type == pygame.MOUSEBUTTONUP:
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
                            # TODO: Might still be duplicate, hard to check for
                            # duplicate edges

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

        if event.type == pygame.MOUSEMOTION:
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

    # update sprites
    nodeSprites.update()
    edgeSprites.update()
    if debug:
        debugCriticalPointsSprites = pygame.sprite.RenderPlain(
                debugCriticalPoints)
        debugDeleteLinesSprites = pygame.sprite.RenderPlain(debugDeleteLines)
        debugSprites.update()
        debugCriticalPointsSprites.update()
        debugDeleteLinesSprites.update()

    # draw to the screen
    screen.blit(background, (0, 0))
    edgeSprites.draw(screen)
    nodeSprites.draw(screen)
    if debug:
        debugSprites.draw(screen)
        debugCriticalPointsSprites.draw(screen)
        debugDeleteLinesSprites.draw(screen)
    pygame.display.flip()

    # clock.tick() returns milliseconds since last call.
    dt = clock.tick(60) / 1000

pr.disable()
# pr.create_stats()
pr.dump_stats("profile")
pstats.Stats("profile").sort_stats(pstats.SortKey.CUMULATIVE, pstats.SortKey.CALLS).print_stats()
pstats.Stats("profile").sort_stats(pstats.SortKey.CUMULATIVE, pstats.SortKey.CALLS).print_callers()
pstats.Stats("profile").sort_stats(pstats.SortKey.TIME, pstats.SortKey.CALLS).print_callers()
pygame.quit()
