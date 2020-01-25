from Commons.Point import Point

class Sensor:
    def __init__(self, loc:Point, height, cost: float, std: float):
        self.loc, self.height, self.rp, self.cost, self.std = loc, height, -float('inf'), cost, std