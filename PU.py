from Commons.Point import Point
from collections import namedtuple
import math
from random import *


class PUR:
    rloc = namedtuple('rloc', ('r', 'theta'))  # relative location to pu

    def __init__(self, location, received_power: float=None, threshold: float=-90, beta: float=2):
        self.loc = self.rloc(r=location[0], theta=location[1])
        self.thr = threshold
        self.beta = beta
        self.rp = received_power  # power received from its own pu
        self.irp = None           # power received from other pus


class PU:
    def __init__(self, location:Point, n, pur_threshod, pur_beta, pur_dist, power):
        self.loc = location
        self.n = n
        self.pur_threshold = pur_threshod
        self.pur_beta = pur_beta
        if len(pur_dist) == 2:
            self.pur_dist_min, self.pur_dist_max = pur_dist[0], pur_dist[1]
        else:
            self.pur_dist_min = self.pur_dist_max = pur_dist
        # self.pur_dist_min = pur_dist
        self.purs = []
        self.p = power
        self.create_purs()

    def create_purs(self):
        angle = float(360/self.n)
        for i in range(self.n):
            self.purs.append(PUR(location=(uniform(self.pur_dist_min, self.pur_dist_max),
                                           i * math.radians(angle)),
                                 threshold=self.pur_threshold, beta=self.pur_beta))


if __name__ == "__main__":
    pu1 = PU(Point(5,5), 10, 13,2, 10)
    for i in range(pu1.n):
        print(str(i+1), "(", pu1.pur[i].loc.get_cartesian[0], ",", pu1.pur[i].loc.get_cartesian[1],")", pu1.loc.distance(pu1.pur[i].loc))
