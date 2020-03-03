# from typing import List
from MLSpectrumAllocation.PU import *
from MLSpectrumAllocation.SU import *
from typing import List
from MLSpectrumAllocation.commons import *
import multiprocessing
import sys
# import random as rd


class Field:
    def __init__(self, pus: List[PU], su: SU, ss:List[Sensor], corners: List[Point], propagation_model: str, alpha: float, noise: bool=False,
                 std: float = -float('inf'), splat_upper_left=None, cell_size: int = 1):
        self.pus = pus
        self.su = su
        self.corners = corners
        # self.propagation_model = propagation_model
        # self.alpha = alpha
        if propagation_model.lower() == 'log':
            self.propagation_model = PropagationModel('log', [alpha])
        elif propagation_model.lower() == 'splat':
            self.propagation_model = PropagationModel('splat', [splat_upper_left])

        self.noise = noise
        self.cell_size = cell_size
        self.std = std
        self.ss = ss
        # self.compute_purs_powers()
        # self.compute_sss_received_power() if ss else None


    def compute_field_power(self) -> List[List[float]]:  # calculate received power at specific locations all over the
                                                            # field to create a heatmap
        received_power = []
        [min_x, min_y] = self.corners[0].get_cartesian
        [max_x, max_y] = self.corners[-1].get_cartesian
        for y in range(min_y, max_y, min(1, max(1,(max_y - min_y) // 400))):
            tmp = []
            for x in range(min_x, max_x, min(1, max(1, (max_x - min_x) // 400))):
                pwr_tmp = -float('inf')
                for pu in self.pus:
                    pwr_tmp = power_with_path_loss(tx=TRX(pu.loc, pu.p, pu.height), rx=TRX(Point(x, y), pwr_tmp, 15),
                                                   propagation_model=self.propagation_model,
                                                   noise=self.noise, cell_size=self.cell_size)
                tmp.append(pwr_tmp)
            received_power.append(tmp)
        return received_power

    def compute_purs_powers(self):  # calculate received powers at PURs
        for pu in self.pus:
            for pur in pu.purs:
                pur_location = pu.loc.add_polar(pur.loc.r, pur.loc.theta)
                pur.rp = -float('inf')  # calculate power received at PURs due to their pu
                pur.rp = power_with_path_loss(tx=TRX(pu.loc, pu.p, pu.height), rx=TRX(pur_location, pur.rp, pur.height),
                                              propagation_model=self.propagation_model,
                                              noise=self.noise, std=self.std, cell_size=self.cell_size)  # , self.noise)
                pur.irp = -float('inf')
                # multiprocessing to increase speed
                # try:
                #     with multiprocessing.Pool() as pool:
                #         jobs = pool.imap_unordered(power_with_path_loss, [(TRX(npu.loc, npu.p, npu.height),
                #                                                     TRX(pur_location, -float('inf'), pur.height),
                #                                                     self.propagation_model, self.noise, self.std)
                #                                                    for npu in self.pus if npu != pu])
                #         pool.close()
                #     irps_res = jobs
                # except Exception as e:
                #     e = sys.exc_info()[0]
                #     print(e)
                #     raise
                # total_irp_power = sum([10 ** (irp/10) for irp in irps_res])
                # pur.irp = 10 * math.log10(total_irp_power) if total_irp_power > 0 else -float('inf')
                for npu in self.pus:  # power received from other PUs
                    if pu != npu:
                        pur.irp = power_with_path_loss(tx=TRX(npu.loc, npu.p, npu.height), rx=TRX(pur_location, pur.irp, pur.height),
                                                       propagation_model=self.propagation_model,
                                                       noise=self.noise, std=self.std, cell_size=self.cell_size)  # , self.noise)

    def compute_sss_received_power(self):  # compute received power at sensors from PUs
        for sensor in self.ss:
            sensor.rp = - float('inf')
            for pu in self.pus:
                sensor.rp = power_with_path_loss(tx=TRX(pu.loc, pu.p, pu.height), rx=TRX(sensor.loc, sensor.rp, sensor.height),
                                                 propagation_model=self.propagation_model, noise=self.noise,
                                                 std=self.std, cell_size=self.cell_size)

    def su_request_accept(self, sign: bool=True):  # sign here is temporary and just for create an unreal situation in which learning would fail
        option = 0  # 0 means using BETA, 1 means using threshold
        for pu in self.pus:
            for pur in pu.purs:
                pur_location = pu.loc.add_polar(pur.loc.r, pur.loc.theta)
                if option:
                    if pur.irp < pur.thr:
                        if power_with_path_loss(tx=TRX(self.su.loc, self.su.p, self.su.height),
                                                rx=TRX(pur_location, pur.irp, pur.height),
                                                propagation_model=self.propagation_model,
                                                noise=self.noise, std=self.std, cell_size=self.cell_size) > pur.thr:
                            return False
                else:
                    try:
                        if 10 ** (pur.irp/10) == 0 or 10 ** (pur.rp/10) / 10 ** (pur.irp/10) > pur.beta:
                            if (10 ** (pur.rp/10) /
                                10 ** (power_with_path_loss(tx=TRX(self.su.loc, self.su.p, self.su.height),
                                                            rx=TRX(pur_location, pur.irp, pur.height),
                                                            propagation_model=self.propagation_model,
                                                            noise=self.noise, std=self.std, cell_size=self.cell_size)/10))\
                                    < pur.beta:
                                return False
                    except ZeroDivisionError as err:
                        print(err, ' happened')
        return True

    # def power_with_path_loss(self, tx:Point, rx:Point, tx_power, rx_power, noise: bool=False, sign: bool=True):  #False for sign means negative otherwise positive
    #     if self.propagation_model.lower() == "log":
    #         loss = 0 if tx.distance(rx) < 1 else 10 * self.propagation_model.var[0] * math.log10(tx.distance(rx))
    #         if noise:
    #             noise = gauss(0, 10 ** (self.std/10))
    #             noise = abs(noise) if sign else -abs(noise)
    #             tx_noisy = 10 ** (tx_power/10) + noise
    #             tx_power = 10 * math.log10(tx_noisy) if tx_noisy > 0 else -float('inf')
    #         res = 10 ** (rx_power / 10) + 10 ** ((tx_power - loss) / 10)
    #         return float(10 * math.log10(res)) if res > 0 else -float('inf')

