import random as rd
import os
if __name__ == '__main__':

    # sensor_file_path = 'rsc/sensors/1000/1200/sensors'
    STYLE = "UNIFORM"  # {"RANDOM", "UNIFOMR"}
    grid_length = 1000
    std = 1
    cost = 0.388882393442197
    number_of_sensors = 3600
    sensor_file_path = '/'.join(['rsc', 'sensors', str(grid_length), str(number_of_sensors)])
    print(sensor_file_path)
    if not os.path.exists(sensor_file_path):
        os.makedirs(sensor_file_path)
    with open(sensor_file_path + '/sensors', 'w') as f:
        if STYLE == "random":
            for i in range(number_of_sensors):
                # x = round(rd.uniform(0, grid_length),2)
                # y = round(rd.uniform(0, grid_length), 2)
                x = round(rd.randint(0, grid_length - 1), 2)
                y = round(rd.randint(0, grid_length - 1), 2)
                f.write(str(x) + " " + str(y) + " " + str(std) + " " + str(cost) + "\n")
        elif STYLE == "UNIFORM":
            row_col_num = int(number_of_sensors ** 0.5)
            distance = grid_length / row_col_num
            points = [p * distance for p in range(row_col_num + 1)]
            sensor_points = [min(int((points[i] + points[i + 1]) / 2), grid_length - 1) for i in range(row_col_num)]
            sensor_locations = set([(x, y) for x in sensor_points for y in sensor_points])
            for x in sensor_points:
                for y in sensor_points:
                    f.write(str(x) + " " + str(y) + " " + str(std) + " " + str(cost) + "\n")
            # REST number_of_sensors - row_col_num ** 2 would be random
            sensor_cnt = row_col_num ** 2
            while sensor_cnt < number_of_sensors:
                x = round(rd.randint(0, grid_length - 1), 2)
                y = round(rd.randint(0, grid_length - 1), 2)
                if (x, y) not in sensor_locations:
                    sensor_cnt += 1
                    sensor_locations.add((x, y))
                    f.write(str(x) + " " + str(y) + " " + str(std) + " " + str(cost) + "\n")
    f.close()