# metering.py
# flir_image --> get temperatures and load to db


import os
import glob
import datetime
import logging
import subprocess
import shutil

import cv2 as cv

from my_utils import Debug, KeyPoint, Misc
from config import Config
from qr_read import QrDecode
from flir_image import FlirImage
from db import Db
import colors

logger = logging.getLogger('thermo.'+'metering')


class Cfg(Config):
    inp_folder = f'../data/tests/groups/2/'  #
    inp_fname_mask = f'*.jpg'  # 512 0909 3232 446
    csv_file = f'../tmp/metering.csv'
    log_folder = f'../tmp/res_preproc/'
    log_file = f'../tmp/debug.log'
    sync_folder = f'/home/im/mypy/thermo/GDrive-Ihorm/FLIR'
    # log_level = logging.DEBUG  # INFO DEBUG WARNING
    log_image = True
    need_sync = True  # sync: run rclone, then move from .../FLIR to inp_folders
    need_csv = True  # save temperature to csv file
    sync_cmd = f'rclone move remote: {os.path.split(sync_folder)[0]}'  # remove /FLIR, it will be added by rclone


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

    def save_to_csv(self):
        with open(Cfg.csv_file, 'a') as f:
            dtime = Misc.dtime_to_csv_str(self.flir_image.datetime)
            f.write(f'{dtime}\t{self.meter_id}\t{self.temperature:.2f}\n')

    def draw_temp(self, image, xy):
        cv.circle(image, xy, 10, colors.BGR_GREEN, 2)
        cv.putText(image, f'{self.meter_id}:{self.temperature:.1f}', xy,
                   cv.FONT_HERSHEY_SIMPLEX, 1, colors.BGR_GREEN, 3)

    def draw_visual(self, image):
        self.draw_temp(image, self.visual_xy)

    def draw_thermal(self, image):
        # thermal_normalized = (thermal_np - np.amin(thermal_np)) / (np.amax(thermal_np) - np.amin(thermal_np))
        # img_thermal = np.array(np.uint8(cm.inferno(thermal_normalized) * 255))  # inferno,gray
        self.draw_temp(image, self.thermo_xy)

    def read_temperature(self, offset, qr_mark, fi):
        anchor = qr_mark.anchor
        visual_xy = KeyPoint.offset_to_xy(offset, anchor)
        if not KeyPoint.check_xy_bound(visual_xy, fi.visual_img):
            logger.error(f'offset({offset})-->{visual_xy} which is not in range for mark {qr_mark.code}'
                         f'for visual image of {fi.fname_path}')
            return None, None, None
        thermo_xy = fi.visual_xy_to_therm(visual_xy)
        if not KeyPoint.check_xy_bound(thermo_xy, fi.thermal_np):
            logger.error(f'visual_xy_to_therm({visual_xy}) --> {thermo_xy} '
                         f'which is not in range for thermal image of {fi.fname_path}')
            return None, None, None
        logger.debug(f'read temper:  {offset}, {visual_xy}, {thermo_xy}')
        temperature = fi.point_temperature(thermo_xy)
        return temperature, visual_xy, thermo_xy

    def __str__(self):
        return (f'file:{self.flir_image.fname_path} code={self.qr_mark.code} meter={self.meter_id} '
                f'offs={self.offset} vis_xy={self.visual_xy} therm_xy= {self.thermo_xy} '
                f'temper={self.temperature}')


def take_readings(fname_path_flir):
    # file --> db + files in images/date
    try:
        fi = FlirImage(fname_path_flir)
    except ValueError:
        logger.error(f'file {fname_path_flir} skipped due to ValueError')
        return 0
    qr_mark_list = QrDecode.get_all_qrs(fi.visual_img)

    visual_img_copy = fi.visual_img.copy()
    thermal_img_copy = fi.thermal_img.copy()

    reading_cnt = 0
    for qr_mark in qr_mark_list:
        meter_records = Db.get_meters_from_db(qr_mark.code)
        if meter_records is None:
            logger.info(f'qrs_to_readings: no meters for qr_mark {qr_mark}')
            continue
        for meter_id, offset_x, offset_y in meter_records:
            if offset_x == 9999.:
                continue
            logger.debug(f'take readings: meter_id={meter_id}: ')

            reading = Reading(meter_id, (offset_x, offset_y), qr_mark, fi)

            if reading.temperature is None:
                logger.error(f'cannot create Reading '
                             f'for meter_id={meter_id} offset=({offset_x},{offset_y}) '
                             f'due to illegal coordinates after offset_to_xy()')
                continue
            reading.save_to_db()
            reading.save_to_csv()

            logger.info(reading)
            reading.draw_visual(visual_img_copy)
            reading.draw_thermal(thermal_img_copy)
            Debug.log_image('visual_read', visual_img_copy)
            Debug.log_image('thermal_read', thermal_img_copy)
            reading_cnt += 1
    return reading_cnt

def sync_meterings():
    os.makedirs(Cfg.inp_folder,exist_ok=True)
    logger.debug(f'sync started: sync_folder={Cfg.sync_folder} inp_folder={Cfg.inp_folder}')
    logger.debug(f'Files: {len(os.listdir(Cfg.sync_folder))} in sync_folder, '
                 f'{len(os.listdir(Cfg.inp_folder))} in inp_folder')
    logger.debug(f'sync metering, cmd={Cfg.sync_cmd}')
    os.system(Cfg.sync_cmd)
    logger.debug(f'Files moved Gdrive --> sync_folder. '
                 f'Files: {len(os.listdir(Cfg.sync_folder))} in sync_folder, '
                 f'{len(os.listdir(Cfg.inp_folder))} in inp_folder')

    for f in glob.glob(Cfg.sync_folder+'/*'):
        print(f'f={f}, dst={Cfg.inp_folder}')
        shutil.move(f,Cfg.inp_folder)
    logger.debug(f'Files moved sync_folder --> input_folder. '
                 f'Files: {len(os.listdir(Cfg.sync_folder))} in sync_folder, '
                 f'{len(os.listdir(Cfg.inp_folder))} in inp_folder')


def main():
    logger.debug('metering - start')
    Debug.set_params(log_folder=Cfg.log_folder, log_image=Cfg.log_image)
    if Cfg.need_sync:
        sync_meterings()
    if Cfg.need_csv:
        with open(Cfg.csv_file, 'w') as f:
            f.write(f'datetime\tmeter_id\ttemperature\n')
    # Db.connect()
    start = datetime.datetime.now()

    files_cnt = 0
    for folder in sorted(glob.glob(f'{Cfg.inp_folder}')):
        if not os.path.isdir(folder):
            continue

        files_list = sorted(glob.glob(f'{folder}/{Cfg.inp_fname_mask}'))
        for files_cnt in range(len(files_list)):
            fname_path_flir = files_list[files_cnt]
            Debug.set_log_image_names(fname_path_flir)

            cnt = take_readings(fname_path_flir)

            print(f'{files_cnt+1} of {len(files_list)}: {fname_path_flir} \t\t\t read_cnt={cnt}')

        # print(f'Folder {folder}: {files_cnt} files processed')

    seconds = (datetime.datetime.now() - start).seconds
    if not files_cnt:
        print(f'no files processed. folders={Cfg.inp_folder} files mask={Cfg.inp_fname_mask}')
    else:
        print(f'Total time = {seconds:.0f}s   average={seconds/(files_cnt+1):.0f}s per file')

    Db.close()


if __name__ == '__main__':
    main()
