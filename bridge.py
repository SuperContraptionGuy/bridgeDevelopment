class Node():
    def __init__(self, location=(0, 0), type="movable"):
        self.location = location
        self.type = type


class Edge:
    def __init__(self, parents=(Node(), Node()), type="support"):
        self.parents = parents
        self.type = type
        match type:
            case "support":
                self.maxLength = 75
                self.strength = 1
                self.color = (255, 251, 40)
                self.thickness = 4
            case "road":
                self.maxLength = 75
                self.strength = 1
                self.color = (8, 8, 10)
                self.thickness = 5
            case "cable":
                self.maxLength = 200
                self.strength = 1
                self.color = (184, 188, 188)
                self.thickness = 2
            case _:
                raise ValueError("Edge type not suporrted")
