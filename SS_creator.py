import random as rd
if __name__ == '__main__':

    sensor_file_path = 'rsc/sensors/1000/1200/sensors'
    grid_length = 1000
    std = 1
    cost = 0.388882393442197
    number_of_sensors = 1200
    with open(sensor_file_path, 'w') as f:
        for i in range(number_of_sensors):
            # x = round(rd.uniform(0, grid_length),2)
            # y = round(rd.uniform(0, grid_length), 2)
            x = round(rd.randint(0, grid_length - 1), 2)
            y = round(rd.randint(0, grid_length - 1), 2)
            f.write(str(x) + " " + str(y) + " " + str(std) + " " + str(cost)+ "\n")
    f.close()