
from Commons.Point import Point


class SU:    # secondary user
    def __init__(self, location:Point, height,  power:float):
        self.loc = location
        self.p = power
        self.height = height