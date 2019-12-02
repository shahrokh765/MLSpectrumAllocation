import random as rd
if __name__ == '__main__':

    sensor_file_path = 'rsc/50/sensors'
    grid_length = 4000
    std = 1
    cost = 0.388882393442197
    number_of_sensors = 50
    with open(sensor_file_path, 'w') as f:
        for i in range(number_of_sensors):
            x = rd.uniform(0, grid_length)
            y = rd.uniform(0, grid_length)
            f.write(str(x) + " " + str(y) + " " + str(std) + " " + str(cost)+ "\n")
    f.close()