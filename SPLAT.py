from collections import namedtuple
import os, sys
import urllib.request
import zipfile
import shutil
import glob
import subprocess
from random import randint
import re
import math
from mpu import haversine_distance

from MLSpectrumAllocation.splat_site import Site

loc = namedtuple('loc', ('lat', 'lon'))

class SPLAT:
    SDF_DIR = 'rsc/splat/sdf'
    OFFSET = loc(-1/111111, -1/111111)  # offset for one meter that should be added to upper_left_loc

    def __init__(self, upper_left_loc: tuple, field_width: int, field_length: int):
        # field_width => x
        # filed_length => y
        self.upper_left_loc = loc(upper_left_loc[0], upper_left_loc[1])  # (lat, lon) of upper left corner
        self.field_width = field_width
        self.field_length = field_length
        self.create_lrp_file()

    def create_lrp_file(self):
        LRP = "25.000  ;   Earth Dielectric Constant (Relative permittivity)\n" + \
              "0.020   ;   Earth Conductivity (Siemens per meter)           \n" + \
              "301.000 ;   Atmospheric Bending Constant (N-units)           \n" + \
              "600.000 ;   Frequency in MHz (20 MHz to 20 GHz)              \n" + \
              "5       ;   Radio Climate (5 = Continental Temperate)        \n" + \
              "1       ;   Polarization (0 = Horizontal, 1 = Vertical)      \n" + \
              "0.50    ;   Fraction of situations (50% of locations)        \n" + \
              "0.90    ;   Fraction of time (90% of the time)\n"

        with open(self.SDF_DIR + '/splat.lrp', 'w') as fq:
            fq.write(str(LRP))


    def generate_sdf_files(self):  # downloading and creating sdf(terrain)
        if not os.path.exists(self.SDF_DIR):
            os.makedirs(self.SDF_DIR)
        tmp_dir = self.SDF_DIR + '/tmp'
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        # if not os.path.exists(self.SDF_DIR + '/out'):
        #     os.makedirs(self.SDF_DIR + '/out')

        lat_set = set([int(self.upper_left_loc.lat), int(SPLAT.get_loc(self.upper_left_loc, 0, self.field_length).lat)])
        lon_set = set([int(self.upper_left_loc.lon), int(SPLAT.get_loc(self.upper_left_loc, self.field_length, 0).lon)])

        for lat in lat_set:
            for lon in lon_set:
                lat_str, lon_str = str(lat), str(lon+1)
                if len(lon_str) < 3:
                    lon_str = '0' + lon_str
                sdf_file = "N" + lat_str + "W" + lon_str + ".hgt.zip"
                print("Downloading Terrain file: ", sdf_file)
                terrain_file_url = "https://dds.cr.usgs.gov/srtm/version2_1/SRTM3/North_America/" + sdf_file
                try:
                    with urllib.request.urlopen(terrain_file_url) as response, open(
                            tmp_dir + '/' + str(sdf_file), 'wb') as f:
                        shutil.copyfileobj(response, f)
                except IOError as e:
                    raise ("Error: terrain file " + sdf_file + " NOT found!", e)
                print('Unzipping SDF file', sdf_file)
                try:
                    # ---uncompress the zip file-----------------#
                    zip_ref = zipfile.ZipFile(tmp_dir + "/" + str(sdf_file), 'r')
                    zip_ref.extractall(tmp_dir)
                    zip_ref.close()
                except (zipfile.BadZipFile, zipfile.LargeZipFile) as e:
                    raise ("Error: Unzipping for", sdf_file, ' was NOT successful!', e)
        print("Downloading and unzipping was successful.")

        # convert zip file to hgt_file using strm2sdf command line
        owd = os.getcwd()
        os.chdir(self.SDF_DIR)
        pp = os.getcwd()
        try:
            for hgt_file in glob.glob('./tmp/*.hgt'):
                subprocess.call(["srtm2sdf", hgt_file])
                # subprocess.call(["srtm2sdf-hd", hgt_file])
        except (ValueError, subprocess.CalledProcessError, OSError) as e:
            e = sys.exc_info()[0]
            raise("Error: converting .hgt file", hgt_file, "was NOT successful!", e)
        os.chdir(owd)
        shutil.rmtree(tmp_dir)  # remove the temporary directory created at the beginning

    @staticmethod
    def get_loc(ref: loc, x, y) -> loc:  # get location in (lat, lon) format when x and y is present
        return loc(ref.lat + y * SPLAT.OFFSET.lat, ref.lon + x * SPLAT.OFFSET.lon /
                   math.cos(math.radians(ref.lat + y * SPLAT.OFFSET.lat)))

    @staticmethod
    def create_qth_files(site: Site):
        qthfile = site.name
        with open(qthfile + '.qth', 'w') as fq:
            fq.write(str(site))
        return qthfile

    @staticmethod
    def path_loss(upper_left_ref: loc, tx: tuple, tx_height: float, rx: tuple, rx_height: float):
        pwd = os.getcwd()
        os.chdir(SPLAT.SDF_DIR)
        # terr_dir = os.getcwd()
        # os.chdir('out')

        tx_loc = SPLAT.get_loc(upper_left_ref, tx[0], tx[1])
        rx_loc = SPLAT.get_loc(upper_left_ref, rx[0], rx[1])
        # print(haversine_distance(tx_loc, rx_loc)*1000)
        tx_site = Site('tx', tx_loc.lat, tx_loc.lon, tx_height)
        rx_site = Site('rx', rx_loc.lat, rx_loc.lon, rx_height)
        tx_name = SPLAT.create_qth_files(tx_site)
        rx_name = SPLAT.create_qth_files(rx_site)

        # running splat command
        path_loss_command = ['splat-hd', '-t', tx_name, '-r', rx_name]
        subprocess.call(path_loss_command, stdout=open(os.devnull, 'wb'))

        output_name = tx_name + '-to-' + rx_name + '.txt'  # the file where the result will be created
        free_pl, itm_pl = SPLAT.process_output(output_name)

        # removing created files
        os.remove(output_name)
        os.remove(tx_name + '.qth')
        os.remove(rx_name + '.qth')
        os.remove(tx_name + '-site_report.txt')

        os.chdir(pwd)
        return float(free_pl), float(itm_pl)

    @staticmethod
    def process_output(file_name):
        positive_float = r'(\d+\.\d+)'
        free_space_pattern = r'Free space.*\D{}.*'.format(positive_float)
        itm_pattern = r'ITWOM Version 3.0.*\D{}.*'.format(positive_float)
        free_p = re.compile(free_space_pattern)
        itm_p = re.compile(itm_pattern)
        with open(file_name, encoding="ISO-8859-1", mode='r') as f:
            content = f.read()
            free_m = free_p.search(content)
            free_pl = free_m.group(1) if free_m else 0

            itm_m = itm_p.search(content)
            itm_pl = itm_m.group(1) if itm_m else 0
        return free_pl, itm_pl




if __name__ == "__main__":
    # 40.800595, -73.107507
    top_left_ref = (40.800595, 73.107507)
    splat = SPLAT(top_left_ref, 1000, 1000)
    # splat.generate_sdf_files()
    free_pl, itm_pl = splat.path_loss(splat.upper_left_loc, (180, 170), 300, (100, 100), 15)
    print('free path loss:', free_pl, ', itm path loss:', itm_pl)