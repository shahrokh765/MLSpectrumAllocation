from Commons.Point import Point

class Sensor:
    def __init__(self, loc:Point, cost: float, std: float):
        self.loc, self.rp, self.cost, self.std = loc, -float('inf'), cost, std

if __name__ == '__main__':
    p = Point(0, 1)
    print(p.get_cartesian())
