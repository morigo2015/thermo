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
    inp_folders = f'../data/tests/door-ideal-rotate'  #
    inp_fname_mask = f'*546.jpg'  # 546
    log_folder = f'../tmp/res_preproc'
    verbose = 0


class Reading:

    def __init__(self, meter_id, offset, qr_mark, flir_image):
        self.meter_id = meter_id
        self.flir_image = flir_image
        self.offset = offset
        self.qr_mark = qr_mark
        self.temperature, self.visual_xy, self.thermo_xy = self.read_temperature(offset, qr_mark, flir_image)


    def save_to_db(self):
        Db.save_reading_to_db(self.flir_image.datetime, self.meter_id,
                              self.flir_image.image_id, self.temperature)

    def draw_temp(self,image,xy):
        cv.circle(image, xy, 10, colors.BGR_GREEN, 2)
        cv.putText(image, f'{self.meter_id}:{self.temperature:.1f}', xy,
                   cv.FONT_HERSHEY_SIMPLEX, 1, colors.BGR_GREEN, 3)

    def draw_visual(self, image):
        self.draw_temp(image,self.visual_xy)

    def draw_thermal(self, image):
        # thermal_normalized = (thermal_np - np.amin(thermal_np)) / (np.amax(thermal_np) - np.amin(thermal_np))
        # img_thermal = np.array(np.uint8(cm.inferno(thermal_normalized) * 255))  # inferno,gray
        self.draw_temp(image,self.thermo_xy)

    def read_temperature(self, offset, qr_mark, fi):
        anchor = qr_mark.anchor
        visual_xy = KeyPoint.offset_to_xy(offset, anchor)
        if not KeyPoint.check_xy_bound(visual_xy,fi.visual_img):
            Debug.error(f'offset({offset})-->{visual_xy} which is not in range '
                        f'for visual image of {fi.fname_path}')
            return None, None, None
        thermo_xy = fi.visual_xy_to_therm(visual_xy)
        if not KeyPoint.check_xy_bound(thermo_xy,fi.thermal_np):
            Debug.error(f'visual_xy_to_therm({visual_xy}) --> {thermo_xy} '
                        f'which is not in range for thermal image of {fi.fname_path}')
            return None, None, None
        Debug.print(f'read temper:  {offset}, {visual_xy}, {thermo_xy}',verbose_level=2)
        temperature = fi.point_temperature(thermo_xy)
        return temperature, visual_xy, thermo_xy

    def __str__(self):
        return (f'file:{self.flir_image.fname_path} code={self.qr_mark.code} meter={self.meter_id} '
                f'offs={self.offset} vis_xy={self.visual_xy} therm_xy= {self.thermo_xy} '
                f'temper={self.temperature}')


def take_readings(fname_path_flir):
    # file --> db + files in images/date
    fi = FlirImage(fname_path_flir)
    qr_mark_list = QrDecode.get_all_qrs(fi.visual_img)

    visual_img_copy = fi.visual_img.copy()
    thermal_img_copy = fi.thermal_img.copy()

    for qr_mark in qr_mark_list:
        meter_records = Db.get_meters_from_db(qr_mark.code)
        if meter_records is None:
            Debug.print(f'qrs_to_readings: no meters for qr_mark {qr_mark}')
            continue
        for meter_id, offset_x, offset_y in meter_records:
            if offset_x == 9999. :
                continue
            Debug.print(f'take readings: meter_id={meter_id}: ',verbose_level=2)
            reading = Reading(meter_id, (offset_x, offset_y), qr_mark, fi)
            if reading.temperature is None:
                Debug.error(f'cannot create Reading '
                            f'for meter_id={meter_id} offset=({offset_x},{offset_y}) '
                            f'due to illegal coordinates after offset_to_xy()')
                continue
            Debug.print(reading)
            reading.draw_visual(visual_img_copy)
            reading.draw_thermal(thermal_img_copy)
            reading.save_to_db()
            Debug.log_image('visual_read', visual_img_copy)
            Debug.log_image('thermal_read', thermal_img_copy)



def main():
    shutil.rmtree(Cfg.log_folder, ignore_errors=True)
    os.makedirs(Cfg.log_folder, exist_ok=True)
    Debug.set_params(log_folder=Cfg.log_folder, verbose=Cfg.verbose)
    # Db.connect()

    for folder in sorted(glob.glob(f'{Cfg.inp_folders}')):
        if not os.path.isdir(folder):
            continue

        files_cnt = 0
        for fname_path_flir in glob.glob(f'{folder}/{Cfg.inp_fname_mask}'):
            Debug.set_params(input_file=fname_path_flir)

            take_readings(fname_path_flir)

            files_cnt += 1
            print(f'{fname_path_flir}')

        print(f'Folder {folder}: {files_cnt}')

    Db.close()


if __name__ == '__main__':
    main()

    #
    # class Meter:
    #
    #     def __init__(self, meter_id, offset):
    #         self.meter_id = meter_id
    #         self.offset = offset

    # def draw_flir(self):
    #
    #     self.draw( self.flir_image.visual_img )
    #
    #     for img, suffix in [(self.flir_image.visual_img, 'vis'),
    #                           (self.flir_image.flir_img, 'flr')]:
    #         image = img.copy()
    #         self.draw(image)
    #         fname = f'{Cfg.log_folder}/' \
    #                 f'{os.path.basename(self.flir_image.fname_path)[:-4]}_' \
    #                 f'{self.meter.meter_id}_{suffix}.jpg'
    #         cv.imwrite(fname, image)

    # Reading.draw_all_readings(fi,readings)
    # for reading in readings:
    #     reading.save_to_db()
