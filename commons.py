# from MLSpectrumAllocation.Field import Field
import math
from random import gauss
from collections import namedtuple
from MLSpectrumAllocation.Sensor import *
from typing import List
import random as rd
from MLSpectrumAllocation.PU import *
from MLSpectrumAllocation.SU import SU

TRX = namedtuple('TRX', ('loc', 'pow'))
PropagationModel = namedtuple('PropagationModel', ('name', 'var'))


def calculate_max_power(pus: List[PU], su: SU, propagation_model: PropagationModel) -> float:
    max_pow = float('inf')
    for pu in pus:
        for pur in pu.purs:
            pur_location = pu.loc.add_polar(pur.loc.r, pur.loc.theta)
            su_power_at_pur = 10 ** (pur.rp/10) / pur.beta - 10 ** (pur.irp/10)
            if su_power_at_pur <= 0:
                return -float('inf')
            if propagation_model.name.lower() == "log":
                loss = 0 if pur_location.distance(su.loc) < 1 else 10 * propagation_model.var[0] * \
                                                                   math.log10(pur_location.distance(su.loc))
                su_power_at_su = 10 * math.log10(su_power_at_pur) + loss
                max_pow = min(max_pow, su_power_at_su)
    return max_pow


# calculate power at receiver if transmitter send a power
def power_with_path_loss(tx:TRX, rx:TRX, propagation_model: PropagationModel, noise: bool=False, std: float=0.0,
                         sign: bool=True):  #False for sign means negative otherwise positive
    if propagation_model.name.lower() == "log":
        tx_power = tx.pow
        loss = 0 if tx.loc.distance(rx.loc) < 1 else 10 * propagation_model.var[0] * math.log10(tx.loc.distance(rx.loc))
        if noise:
            noise = gauss(0, 10 ** (std/10))
            noise = abs(noise) if sign else -abs(noise)
            tx_noisy = 10 ** (tx_power/10) + noise
            tx_power = 10 * math.log10(tx_noisy) if tx_noisy > 0 else -float('inf')
        res = 10 ** (rx.pow / 10) + 10 ** ((tx_power - loss) / 10)
        return float(10 * math.log10(res)) if res > 0 else -float('inf')

# Create sensors from a file
def create_sensors(path:str)-> List[Sensor]:
    SS = []
    try:
        with open(path, 'r') as f:
            # max_gain = 0.5 * num_intruders  # len(self.transmitters)
            # index = 0
            lines = f.readlines()
            for line in lines:
                line = line.split(' ')
                x, y, std, cost = float(line[0]), float(line[1]), float(line[2]), float(line[3])
                SS.append(Sensor(loc= Point(x, y), cost=cost, std=std))
    except FileNotFoundError:
        print('Sensor file does not exist')
    return SS


# compute power SU can send based on conservative model: if existing power at su location is higher than
# noise floor, it cannot send otherwise it can send a power based on propagation model and minimum power a PU can have
def conservative_model_power(pus: List[PU], su: SU, min_power, propagation_model: PropagationModel, noise_floor=-90,
                             noise=False):
    send_power = float('inf')
    for pu in pus: # upper bound should be calculated based on purs not pus. Assume there is minimum power at a PU; should calculate min(pur_pow/beta)
        for pur in pu.purs:
            pur_loc = pu.loc.add_polar(pur.loc.r, pur.loc.theta)
            pow_tmp = power_with_path_loss(tx=TRX(pu.loc, min_power), rx=TRX(pur_loc, -float('inf')),
                                           propagation_model=propagation_model, noise=noise)
            send_power = min(send_power, pow_tmp/pur.beta)
    power_at_su_from_pus = -float('inf')
    for pu in pus:
        power_at_su_from_pus = power_with_path_loss(tx=TRX(pu.loc, pu.p), rx=TRX(su.loc, power_at_su_from_pus),
                                                    propagation_model=propagation_model, noise=noise)
        if power_at_su_from_pus > noise_floor:
            return noise_floor  # there is already a signal, su cannot send anything.
    return send_power  # all powers from pus is still less than threshold, su can send power


# INTERPOLATION
InterSMParam = namedtuple('InterSMParam', ('pu_size', 'ss_size', 'selection_algo', 'gamma'))


def numSelect(select_size, elements_size):
    if select_size == 0 or select_size > elements_size:
        return elements_size
    return select_size


def interpolation_max_power(pus: List[PU], su: SU, sss: List[Sensor], inter_sm_param: InterSMParam,
                            propagation_model:PropagationModel, noise=False):
    k_pu = numSelect(inter_sm_param.pu_size, len(pus))
    k_ss = numSelect(inter_sm_param.ss_size, len(sss))
    if propagation_model.name.lower() == 'log':
        pl_alpha = propagation_model.var[0]

    pu_inds, sss_inds, sss_dists = [], [], []
    if not inter_sm_param.selection_algo:
        pu_inds = list(range(k_pu))
        sss_inds = list(range(k_ss))
        sss_dists = [su.loc.distance(sss[i].loc) for i in range(k_ss)]
    elif inter_sm_param.selection_algo.lower() == 'sort':
        pu_dists = []
        for i, pu in enumerate(pus):
            dist, ind = su.loc.distance(pu.loc), i
            if i < k_pu:
                pu_inds.append(i)
                pu_dists.append(dist)
            else:
                for j in range(len(pu_inds)):
                    if dist < pu_dists[j]:
                        pu_dists[j], dist = dist, pu_dists[j]
                        ind, pu_inds[j] = pu_inds[j], ind

        for i, ss in enumerate(sss):
            dist, ind = su.loc.distance(ss.loc), i
            if i < k_ss:
                sss_inds.append(i)
                sss_dists.append(dist)
            else:
                for j in range(len(sss_inds)):
                    if dist < sss_dists[j]:
                        sss_dists[j], dist = dist, sss_dists[j]
                        ind, sss_inds[j] = sss_inds[j], ind
    elif inter_sm_param.selection_algo.lower() == 'random':
        pu_inds = rd.sample(range(len(pus)), k_pu)
        pu_dists = [su.loc.distance(pus[i].loc) for i in pu_inds]
        sss_inds = rd.sample(range(len(sss)), k_ss)
        sss_dists = [su.loc.distance(sss[i].loc) for i in sss_inds]
    else:
        print('Unsupported selection algorithm!')
        return None
    # end selection

    # compute weights
    weights, tmp_sum_weight = [], 0.0
    for ss_dist in sss_dists:
        d = ss_dist
        d = max(d, 0.0001)  # TODO why 0.0001?

        # Weight of this SS with respect to this SU
        w = (1.0/d) ** pl_alpha
        weights.append(w)
        tmp_sum_weight += w

    # BY ME
    received_powers = []
    pl_pu_ss = []
    for i, ss_idx in enumerate(sss_inds):
        tmp_ss, all_power, tmp_pl_pu_ss = [], 0, []
        for j, pu_idx in enumerate(pu_inds):
            tmp = (10 ** (pus[pu_idx].p/10)) / pus[pu_idx].loc.distance(sss[ss_idx].loc) ** pl_alpha
            tmp_ss.append(tmp)
            all_power += tmp
            tmp_pl_pu_ss.append(10 ** (pus[pu_idx].p/10))
            # tmp_pow = power_with_path_loss(TRX(pus[pu_idx].loc, pus[pu_idx].p), TRX(sss[ss_idx].loc, -float('inf')),
            #                                propagation_model=propagation_model, noise=noise)
            # # tmp_pow = max(tmp_pow, noise_floor)
            # tmp_ss.append(tmp_pow)
        received_powers.append([10 ** (sss[ss_idx].rp/10) / all_power * x for x in tmp_ss])
        pl_pu_ss.append([x/y for x in tmp_pl_pu_ss for y in received_powers[-1]])
    # Compute SU transmit power
    # tp = thresh * sum(w(SS)) / sum(r(SS) / t(PU) * w(SS))
    max_transmit_power, estimated_path_loss = float('inf'), []
    for y, j in enumerate(pu_inds):
        sum_weight = 0.0
        sum_weighted_ratio = 0.0
        for x, i in enumerate(sss_inds):
            sum_weight += weights[x]
            # only DB is implemented here
            sum_weighted_ratio += weights[x] * (received_powers[i][j] / 10 ** (pus[j].p/10))
        this_pu_path_loss = sum_weighted_ratio / sum_weight
        # estimated_path_loss_tmp = []
        for x, pur in enumerate(pus[j].purs):
            pur_location = pus[j].loc.add_polar(pur.loc.r, pur.loc.theta)
            this_pr_path_loss = this_pu_path_loss - 10.0 * inter_sm_param.gamma * \
                                math.log10(su.loc.distance(pur_location) / su.loc.distance(pus[j].loc))
            # estimated_path_loss_tmp.append(this_pr_path_loss)
            # this_transmit_power = pur.thr - this_pr_path_loss
            this_transmit_power = pur.rp/pur.beta - pur.irp + this_pr_path_loss
            max_transmit_power = min(max_transmit_power, this_transmit_power)
        # estimated_path_loss.append(estimated_path_loss_tmp)
    return max_transmit_power

