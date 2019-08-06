# metering.py
# flir_image --> get temperatures and load to db


import os
import shutil
import glob

import cv2 as cv
import numpy as np
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
    log_folder = f'../tmp/res_preproc'
    verbose = 0


class Meter:

    def __init__(self, meter_id, offset):
        self.meter_id = meter_id
        self.offset = offset
        self.temperature = None

    def read_temperature(self, anchor, fi):
        visual_xy = KeyPoint.offset_to_xy(self.offset, anchor)

        img_v = fi.visual_img.copy()
        cv.circle(img_v, visual_xy, 10, colors.BGR_GREEN, 2)
        Debug.log_image('visual', img_v)

        thermo_xy = fi.visual_xy_to_therm(visual_xy)

        thermal_normalized = \
            (fi.thermal_np - np.amin(fi.thermal_np)) / (np.amax(fi.thermal_np)
                                                        - np.amin(fi.thermal_np))
        img_thermal = np.array(np.uint8(cm.inferno(thermal_normalized) * 255))  # inferno,gray
        cv.circle(img_thermal, thermo_xy, 10, colors.BGR_GREEN, 2)
        Debug.log_image('thermal', img_thermal)

        temperature = fi.point_temperature(thermo_xy)
        return temperature


class Reading:

    def __init__(self, flir_image, meter, temperature):
        self.flir_image = flir_image
        self.meter = meter
        self.temperature = temperature

    @staticmethod
    def qrs_to_readings(qr_list, flir_image):
        readings = []
        for qr_mark in qr_list:
            meter_records = Db.get_meters_from_db(qr_mark.code)
            for meter_id, offset in meter_records:
                meter = Meter(meter_id, offset)
                Debug.print(f'file:{flir_image.fname_path} code={qr_mark.code}')
                temperature = meter.read_temperature(qr_mark.anchor, flir_image)
                reading = Reading(flir_image, meter, temperature)
                readings.append(reading)
        return readings

    def save_to_db(self):
        Db.save_reading_to_db(self.flir_image.datetime, self.meter.meter_id,
                              self.flir_image.image_id, self.temperature)


def process_file(fname_path_flir):
    fi = FlirImage(fname_path_flir)
    qr_mark_list = QrDecode.get_all_qrs(fi.visual_img)
    readings = Reading.qrs_to_readings(qr_mark_list, fi)
    for reading in readings:
        reading.save_to_db()


def main():
    shutil.rmtree(Cfg.log_folder, ignore_errors=True)
    os.makedirs(Cfg.log_folder, exist_ok=True)
    Debug.set_params(log_folder=Cfg.log_folder, verbose=Cfg.verbose)

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
