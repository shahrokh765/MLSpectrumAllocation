'''
A splat site that have location(lat, lon) amd height
'''
from random import randint
class Site:
    def __init__(self, type, lat, lon, height):
        self.name = '{}-N{}W{}-{}'.format(type, abs(int(lat*1000)), abs(int(lon*1000)), randint(0, 10**5))
        # self.type = type # 'tx' or 'rx'
        self.lat = lat  # latitude  is the Y axis
        self.lon = lon  # longitude is the X axis
        self.height = height

    def __str__(self):
        return '{}\n{}\n{}\n{}m\n'.format(self.name, abs(self.lat), abs(self.lon), self.height)

if __name__ == "__main__":
    s = Site('tx', 48.68, -65.25, 30)
    print(s)

    s = Site('rx', 48.68, -65.25, 30)
    print(s)