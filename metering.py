# metering.py
# flir_image --> get temperatures and load to db


import os
import shutil
import glob

import cv2 as cv
import numpy as np
from PIL import Image
from matplotlib import cm


from my_utils import Debug, KeyPoint
from config import Config
from qr_read import QrDecode
from flir_image import FlirImage
from db import Db
import colors


class Cfg(Config):
    inp_folders = f'../data/tests/rot*'  #
    inp_fname_mask = f'*013.jpg'
    res_folder = f'../tmp/res_preproc'
    verbose = 2


class Meter:

    def __init__(self, meter_id, offset):
        self.meter_id = meter_id
        self.offset = offset
        self.temperature = None

    def read_temperature(self, anchor, fi):
        visual_xy = KeyPoint.offset_to_xy(self.offset, anchor)

        img_v = fi.visual_np.copy()
        cv.circle(img_v, visual_xy, 10, colors.BGR_GREEN, 2)
        Debug.log_image('visual', img_v)

        thermo_xy = fi.visual_xy_to_therm(visual_xy)

        thermal_normalized = \
            (fi.thermal_np - np.amin(fi.thermal_np)) / (np.amax(fi.thermal_np)
                                                        - np.amin(fi.thermal_np))
        img_thermal = np.array(np.uint8(cm.inferno(thermal_normalized) * 255))  # inferno,gray
        cv.circle(img_thermal, thermo_xy, 10, colors.BGR_GREEN, 2)
        Debug.log_image('thermal', img_thermal)

        self.temperature = fi.point_temperature(thermo_xy)


def process_file(fname_path_flir):
    fi = FlirImage(fname_path_flir)
    qr_mark_list = QrDecode.get_all_qrs(fi.visual_np)
    for qr_mark in qr_mark_list:
        meter_records = Db.get_meters_info(qr_mark.code)
        for meter_id, offset in meter_records:
            Debug.print(f'file:{fname_path_flir} code={qr_mark.code}')
            meter = Meter(meter_id, offset)
            meter.read_temperature(qr_mark.anchor, fi)
            Db.save_meter_value(fi.datetime, meter.meter_id, meter.temperature)


def main():
    shutil.rmtree(Cfg.res_folder, ignore_errors=True)
    os.makedirs(Cfg.res_folder, exist_ok=True)
    Debug.set_params(log_folder=Cfg.res_folder, verbose=Cfg.verbose)

    for folder in sorted(glob.glob(f'{Cfg.inp_folders}')):
        if not os.path.isdir(folder):
            continue

        files_cnt = 0
        for fname_path_flir in glob.glob(f'{folder}/{Cfg.inp_fname_mask}'):
            Debug.set_params(input_file=fname_path_flir)

            process_file(fname_path_flir)

            files_cnt += 1
            print(f'{fname_path_flir}')

        print(f'Folder {folder}: {files_cnt}')


if __name__ == '__main__':
    main()
