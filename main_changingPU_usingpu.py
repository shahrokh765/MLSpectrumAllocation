from MLSpectrumAllocation.Field import *
from random import *
import datetime, time
import numpy as np
import math
from MLSpectrumAllocation.commons import *
from Commons.Point import *
from MLSpectrumAllocation.SPLAT import SPLAT
import tqdm
from multiprocessing import Process, Queue, Manager
import os
import glob
import pickle


def main_parallel(chunk_size, chunk_number, file_appdx, return_dict):
    num_one = 0
    inter_error, inter_ignore = 0, 0
    conserve_error, conserve_ignore = 0, 0
    tmp_f = open("rsc/tmp/pu_" + str(file_appdx) + "_" + str(chunk_number) + ".txt", "w")
    if MAX_POWER:
        tmp_f_max = open("rsc/tmp/max_" + str(file_appdx) + "_" + str(chunk_number) + ".txt", "w")
    if CONSERVATIVE:
        tmp_f_conserve = open("rsc/tmp/conserve_" + str(file_appdx) + "_" + str(chunk_number) + ".txt", "w")
    if ss:
        tmp_f_sensor = open("rsc/tmp/sensor_" + str(file_appdx) + "_" + str(chunk_number) + ".txt", "w")

    pus = []
    for _ in range(pus_number):
        pus.append(PU(location=Point(uniform(0, max_x), uniform(0, max_y)), n=pur_number, pur_threshod=pur_threshold,
                      pur_beta=pur_beta, pur_dist=(min_pur_dist, max_pur_dist), power=uniform(min_power, max_power),
                      height=tx_height, pur_height=rx_height))

    su = SU(location=Point(uniform(0, max_x), uniform(0, max_y)), height=tx_height,
            power=uniform(min_power, max_power))

    field = Field(pus=pus, su=su, ss=ss, corners=corners, propagation_model=propagation_model, alpha=alpha,
                  noise=noise, std=std, splat_upper_left=splat.upper_left_loc)

    for _ in tqdm.tqdm(range(chunk_size)):
        su.loc = Point(uniform(0, max_x), uniform(0, max_y))
        su.p = uniform(max_power - 5, max_power + 55)

        for pu in pus:
            pu.loc, pu.p = Point(uniform(0, max_x), uniform(0, max_y)), uniform(min_power, max_power)

        field.compute_purs_powers()
        field.compute_sss_received_power() if field.ss else None

        res = 0
        if field.su_request_accept(randint(0, 1)):
            res = 1
            num_one += 1
        for pu in pus:
            tmp_f.write(str(pu.loc.get_cartesian[0]) + "," + str(pu.loc.get_cartesian[1]) + "," + str(round(pu.p, 3)) + ",")
        tmp_f.write(
            str(su.loc.get_cartesian[0]) + "," + str(su.loc.get_cartesian[1]) + "," + str(round(su.p, 3)) + "," + str(
                res))
        tmp_f.write("\n")

        if ss:
            for sensor in ss:
                tmp_f_sensor.write(str(round(sensor.rp, 3)) + ",")
            tmp_f_sensor.write(str(su.loc.get_cartesian[0]) + "," + str(su.loc.get_cartesian[1]) + "," +
                               str(round(su.p, 3)) + "," + str(res))
            tmp_f_sensor.write("\n")

        if MAX_POWER:  # used when you want to calculate maximum power of su it can send without any interference
            highest_pow = calculate_max_power(field.pus, field.su, field.propagation_model)

            if CONSERVATIVE:
                conserve_pow = conservative_model_power(pus=field.pus, su=field.su, min_power=min_power,
                                                        propagation_model=field.propagation_model,
                                                        noise_floor=noise_floor,
                                                        noise=noise)
                if conserve_pow > highest_pow:
                    print('Warning: Conservative power higher than max!!!, MAX:', str(highest_pow), ', Conservative: ',
                          str(conserve_pow))
            if INTERPOLARION:
                inter_pow = interpolation_max_power(pus=field.pus, su=field.su, sss=field.ss,
                                                    inter_sm_param=InterSMParam(0, 0, 'sort', 2),
                                                    propagation_model=field.propagation_model, noise=noise)

                if inter_pow > highest_pow:
                    print('Warning: Interpolation power higher than max!!! , MAX:', str(highest_pow),
                          ', Interpolation: ',
                          str(inter_pow))

            if highest_pow != -float('inf'):
                if CONSERVATIVE:
                    conserve_error += abs(highest_pow - conserve_pow)
                if INTERPOLARION:
                    if inter_pow != -float('inf') and inter_pow != float('inf'):
                        inter_error += abs(highest_pow - inter_pow)
                    else:
                        inter_ignore += 1
            else:
                if INTERPOLARION:
                    inter_ignore += 1
                if CONSERVATIVE:
                    conserve_ignore += 1

            tmp_f_max.write(str(su.loc.get_cartesian[0]) + "," + str(su.loc.get_cartesian[1]) + "," +
                        (str(round(highest_pow, 3)) if highest_pow != -float('inf') else '-inf'))
            tmp_f_max.write("\n")
            if CONSERVATIVE:
                tmp_f_conserve.write(str(su.loc.get_cartesian[0]) + "," + str(su.loc.get_cartesian[1]) + "," +
                                 (str(round(highest_pow, 3)) if highest_pow != -float('inf') else '-inf') + "," +
                                 (str(round(conserve_pow, 3)) if conserve_pow != -float('inf') else '-inf') + "," +
                                 (str(round(inter_pow, 3)) if conserve_pow != -float('inf') else '-inf'))
                tmp_f_conserve.write("\n")
    return_dict[chunk_number] = [num_one, (conserve_error, conserve_ignore), (inter_error, inter_ignore)]
    tmp_f.close()
    if MAX_POWER:
        tmp_f_max.close()
    if CONSERVATIVE:
        tmp_f_conserve.close()
    if ss:
        tmp_f_sensor.close()


def write_output(file_name, pattern):
    read_files = glob.glob("rsc/tmp/" + pattern + "*.txt")
    with open(file_name, "wb") as outfile:
        for f in read_files:
            with open(f, "rb") as infile:
                outfile.write(infile.read())
            os.remove(f)


if __name__ == "__main__":
    propagation_model = 'splat'  # 'splat' or 'log'
    alpha = 3  # 2.0, 4.9
    splat_left_upper_ref = (40.800595, 73.107507)
    tx_height, rx_height = 30, 15
    max_x = 200  # in meter
    max_y = 200  # in meter
    corners = [Point(0, 0), Point(max_x, 0), Point(0, max_y), Point(max_x, max_y)]
    pus_number = 10  # number of pus all over the field
    pur_number = 5  # number of purs each pu can have
    pur_threshold = 0
    pur_beta = 1
    pur_dist = 3  # distance from each pur to its pu
    min_power = -30  # in dB
    max_power = 0    # in dB
    min_pur_dist = 1
    max_pur_dist = 3
    noise_floor = -90
    noise, std = False, 30  # std in dB, real std=10^(std/10)
    MAX_POWER = True   # make it true if you want to achieve the highest power su can have without interference.
                        # calculation for conservative model would also be done
    number_of_process = 8
    INTERPOLARION, CONSERVATIVE = False, False

    start_time = time.time()
    splat = SPLAT(splat_left_upper_ref, max_x, max_y)
    # if propagation_model == 'splat':
    #     splat.generate_sdf_files()
    sensors_path = 'rsc/100/sensors'

    if not os.path.exists('rsc/tmp'):
        os.makedirs('rsc/tmp')

    n_samples = 1000

    ss = create_sensors(sensors_path, rx_height)

    date = datetime.datetime.now().strftime('_%Y%m_%d%H_%M')

    jobs = []
    manager = Manager()
    result_dict = manager.dict()

    chunks_size = [n_samples // number_of_process] * number_of_process
    chunks_size[-1] += n_samples % number_of_process

    file_appdx = randint(0, 1000)
    for i in range(number_of_process):
        p = Process(target=main_parallel, args=(chunks_size[i], i, file_appdx, result_dict))
        jobs.append(p)
        p.start()
        # p.join()
    # wait for the jobs to finish
    for job in jobs:
        job.join()

    print('Number of samples of class 1(accepted):', sum([x[0] for x in result_dict.values()]))
    if CONSERVATIVE:
        print('Mean Error for Conservative Model:', sum([x[1][0] for x in result_dict.values()]) /
              (n_samples - sum([x[1][1] for x in result_dict.values()])))
        print('Number of ignore conservative: ', str(sum([x[1][1] for x in result_dict.values()])))
    if INTERPOLARION:
        print('Mean Error for Interpolation Model:', sum([x[2][0] for x in result_dict.values()]) /
              (n_samples - sum([x[2][1] for x in result_dict.values()])))
        print('Number of ignore interpolation: ', str(sum([x[2][1] for x in result_dict.values()])))
    cwd = os.getcwd()

    write_output("ML/data/dynamic_pus_using_pus_" + str(n_samples) + "_" + str(pus_number) + "PUs" +
                 "_" + str(max(max_y, max_x)) + "grid_" + propagation_model +
                 ("_noisy_std" + str(std) if noise else "") + date + ".txt", "pu_" + str(file_appdx))

    if MAX_POWER:
        write_output("ML/data/dynamic_pus_max_power" + str(n_samples) + "_" + str(pus_number) + "PUs"
                     + "_" + str(max(max_y, max_x)) + "grid_" + propagation_model +
                     ("_noisy_std" + str(std)  if noise else "") + date + ".txt", "max_" + str(file_appdx))

    if CONSERVATIVE:
        write_output("ML/data/dynamic_pus_conservative_power" + str(n_samples) + "_" + str(pus_number) + "PUs"
                     + "_" + str(max(max_y, max_x)) + "grid_" + propagation_model +
                     ("_noisy_std" + str(std) if noise else "") + date + ".txt", "conserve_" + str(file_appdx))

    if ss:
        write_output("ML/data/dynamic_pus_sensors_" + str(n_samples) + "_" + str(pus_number) + "PUs" + str(len(ss)) +
                     "_" + str(len(ss)) + "sensors" + "_" + str(max(max_y, max_x)) + "grid_" + propagation_model +
                     ("_noisy_std" + str(std) if noise else "") + date + ".txt", "sensor_" + str(file_appdx))

    if False:   # make it True if you need to find out distribution of location with maximum allowed power
        power_width = max_power - min_power
        # max_allowed_power = [0] * (power_width + 1)
        power_range = np.arange(0, 20, 0.3)
        max_allowed_power = [0] * len(power_range)
        for y in range(max_x):
            for x in range(max_y):
                su.loc = Point(x, y)
                # low = min_power
                # high = max_power
                # mid = (high + low) // 2
                # # while mid - low > 1 and high - mid > 1:
                # while low <= high:
                #     su.p = mid
                #     if field.su_request_accept():
                #         low = mid + 1
                #     else:
                #         high = mid - 1
                #     mid = (high + low) // 2
                # max_allowed_power[mid + power_width] += 1
                for ipow, power in enumerate(power_range):
                    su.p = power
                    if not field.su_request_accept():
                        break
                if ipow != 0 and ipow != len(power_range) - 1:
                    ipow -= 1
                max_allowed_power[ipow] += 1
        print(max_allowed_power)

    if False:  #used for heatmap of the field
        f_heat = open("ML/data/su_only" + str(n_samples) + "_" + str(pus_number) + "PUs" + "_heatmap_" +
                 datetime.datetime.now().strftime('_%Y%m_%d%H_%M') + ".txt", "w")
        received_power = field.compute_field_power()
        for lrp in received_power:  # received power for a specific x=c (all )
            for prp in lrp:         # received pwer for a a specifi (x, y)
                f_heat.write(str(round(prp, 3)) + " ")
            f_heat.write('\n')
        f_heat.close()

    print("--- %s seconds ---" % (time.time() - start_time))
    pl_map_file = open('pl_map_' + str(max(max_x, max_y)) + '.dat', 'w')
    pickle.dump([SPLAT.pl_dict], file=pl_map_file)
    pl_map_file.close()