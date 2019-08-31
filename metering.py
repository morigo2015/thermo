# metering.py
# flir_image --> get temperatures and load to db


import os
import glob
import datetime
import logging
import shutil
import time

import cv2 as cv

from my_utils import Debug, KeyPoint, Misc
from config import Config
from qr_read import QrDecode
from flir_image import FlirImage
from db import Db
from analyzer import Analyzer
import colors

logger = logging.getLogger('thermo.' + 'metering')


class Cfg(Config):
    inp_folder = f'../data/tests/for_demo/d4/'  #
    inp_fname_mask = f'*.jpg'  # 512 0909 3232 446
    csv_file = f'../tmp/metering.csv'
    log_folder = f'../tmp/res_preproc/'
    log_file = f'../tmp/debug.log'
    sync_folder = f'/home/im/mypy/thermo/GDrive-Ihorm/FLIR'
    # log_level = logging.DEBUG  # INFO DEBUG WARNING
    log_image = False
    need_sync = False  # True  # sync: run rclone, then move from .../FLIR to inp_folders
    # need_csv = False  # save temperature to csv file
    sync_cmd = f'rclone move remote: {os.path.split(sync_folder)[0]}'  # remove /FLIR, it will be added by rclone
    purge_reading_flg = False  # readings -> readings_hist
    inp_timeout = 30  # seconds, between attempt to sync/process files
    equip_grp_max_idles = 3  # how many inp_timeouts wait to decide that old equip_id is finished (to call analyzer)


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


class _GroupEquip:
    # detect if equip_id changed in input list (if so - the analyzer should be called)
    # if new readings aren't got during Cfg.equip_group_timeout seconds, then old equip_id is finished
    idle_cycles_left = -1
    current_equip_id = -1

    @classmethod
    def _timeout_is_set_and_expired(cls):
        if cls.wait_cycles_cnt == -1:  # wasn't set
            cls.wait_cycles_cnt = Cfg.equip_grp_max_idles  # set counter
            return False
        cls.wait_cycles_cnt -= 1
        if cls.wait_cycles_cnt == 0:  # was set and is expired now
            cls.wait_cycles_cnt = -1
            return True
        else:
            return False  # not expired yet

    @classmethod
    def ready_to_analyze(cls, event,meter_ids=None):
        # return True if ready to analyze readings (new equip came, or some other cases)
        logger.debug(f'new event:{event}')

        if event == 'readings_taken':
            cls.wait_cycles_cnt = -1  # reset counter since idle cycles are interrupted
            equip_ids = [Db.meter_to_equip(meter_id) for meter_id in list(set(meter_ids))]
            equip_ids_uniq = list(set(equip_ids))
            if len(equip_ids_uniq) == 1: # normal case - only one equip on an image
                if cls.current_equip_id == -1:  # if first image after start
                    cls.current_equip_id = equip_ids_uniq[0]
                if cls.current_equip_id == equip_ids_uniq[0]:
                    return False  # equip is not changed, no need to call analyzer
                else:
                    cls.current_equip_id = equip_ids_uniq[0]
                    return True  # new equip come - call analyzer for old equip
            else:  # several equips on the image
                logger.warning(f'Several equip_id in one image. '
                               f'(Meter,equip):{[(m,Db.meter_to_equip(m)) for m in meter_ids]}')
                if cls.current_equip_id in equip_ids:
                    # ignore new equip_id, we are still near old equip_id
                    return False
                else:
                    # no old equip in image, it's supposed to be new equip
                    # new equip will be the one with max occurrences in equip_ids (non uniq list)
                    cls.current_equip_id = max(equip_ids,key=equip_ids.count)
                    return True

        elif event == 'empty_dir':
            return True if cls._timeout_is_set_and_expired() else False

        elif event == 'the_end':
            cls.wait_cycles_cnt = -1  # just in case
            return True  # run analyzer before closing to process remaining Readings

        else:
            logger.error(f'Illegal event={event}.')
            exit(1)


def take_readings(fname_path_flir):
    # file --> db + files in images/date
    Debug.set_log_image_names(fname_path_flir)

    try:
        fi = FlirImage(fname_path_flir)
    except (ValueError, KeyError):
        logger.exception(f'error while flir-processing file {fname_path_flir}. Skipped.')
        return 0, None
    qr_mark_list = QrDecode.get_all_qrs(fi.visual_img)

    visual_img_copy = fi.visual_img.copy()
    thermal_img_copy = fi.thermal_img.copy()

    reading_cnt = 0
    meter_ids = []
    for qr_mark in qr_mark_list:
        meter_records = Db.get_meters_from_db(qr_mark.code)
        if meter_records is None:
            logger.info(f'qrs_to_readings: no meters for qr_mark {qr_mark}')
            continue
        for meter_id, offset_x, offset_y in meter_records:
            meter_ids.append(meter_id)
            if offset_x == 9999.:
                continue
            logger.debug(f'take readings: meter_id={meter_id}: ')

            reading = Reading(meter_id, (offset_x, offset_y), qr_mark, fi)

            if reading.temperature is None:
                logger.error(f'cannot create Reading '
                             f'for meter_id={meter_id} offset=({offset_x},{offset_y}) '
                             f'due to illegal coordinates after offset_to_xy()')
                continue

            if _GroupEquip.ready_to_analyze(event='readings_taken', meter_ids=meter_ids):
                Analyzer.run('readings_taken')

            reading.save_to_db()
            # reading.save_to_csv()

            # logger.debug(reading)
            reading.draw_visual(visual_img_copy)
            reading.draw_thermal(thermal_img_copy)
            Debug.log_image('visual_read', visual_img_copy)
            Debug.log_image('thermal_read', thermal_img_copy)
            reading_cnt += 1

    return reading_cnt, meter_ids


def sync_meterings():
    os.makedirs(Cfg.inp_folder, exist_ok=True)
    logger.debug(f'sync started: sync_folder={Cfg.sync_folder} inp_folder={Cfg.inp_folder}')
    logger.debug(f'Files: {len(os.listdir(Cfg.sync_folder))} in sync_folder, '
                 f'{len(os.listdir(Cfg.inp_folder))} in inp_folder')
    logger.debug(f'sync metering, cmd={Cfg.sync_cmd}')
    os.system(Cfg.sync_cmd)
    logger.debug(f'Files moved Gdrive --> sync_folder. '
                 f'Files: {len(os.listdir(Cfg.sync_folder))} in sync_folder, '
                 f'{len(os.listdir(Cfg.inp_folder))} in inp_folder')

    for f in glob.glob(Cfg.sync_folder + '/*'):
        print(f'f={f}, dst={Cfg.inp_folder}')
        shutil.move(f, Cfg.inp_folder)
    logger.debug(f'Files moved sync_folder --> input_folder. '
                 f'Files: {len(os.listdir(Cfg.sync_folder))} in sync_folder, '
                 f'{len(os.listdir(Cfg.inp_folder))} in inp_folder')


def readings_to_hist():
    # Readings --> Readings_hist:
    rec_cnt = Db.select('select count(*) from Readings_plus_atmo', (), Db.OneValueRecord)[0].value
    logger.debug(f'copying {rec_cnt} records from view Readings_plus_atmo to table Readings_hist')
    Db.exec('insert into Readings_hist select * from Readings_plus_atmo')

    if Cfg.purge_reading_flg:
        rec_cnt = Db.select('select count(*) from Readings', (), Db.OneValueRecord)[0].value
        logger.debug(f'deleting {rec_cnt} records from Reading')
        Db.exec('delete from Readings')


def main():
    logger.debug('metering - start')
    Debug.set_params(log_folder=Cfg.log_folder, log_image=Cfg.log_image)
    if not os.path.isdir(Cfg.inp_folder):
        logger.error(f'Input folder {Cfg.inp_folder} does not exist.')
        return
    try:
        while True:
            if Cfg.need_sync:
                sync_meterings()
            files_list = sorted(glob.glob(f'{Cfg.inp_folder}/{Cfg.inp_fname_mask}'))
            if not len(files_list):
                if _GroupEquip.ready_to_analyze(event='empty_dir'):
                    Analyzer.run('empty_dir')
                logger.debug(f'timeout {Cfg.inp_timeout} sec')
                time.sleep(Cfg.inp_timeout)  # sec
                continue
            start = datetime.datetime.now()
            for (files_cnt, fname_path_flir) in enumerate(files_list,1):

                cnt, meter_ids = take_readings(fname_path_flir)

                if not cnt or not len(meter_ids):
                    continue  # skip it, no mark/meters/readings here
                logger.info(f'{files_cnt} of {len(files_list)}: {fname_path_flir} readings:{cnt} '
                            f'equip:{list(set([Db.meter_to_equip(m) for m in meter_ids]))}')

            seconds = (datetime.datetime.now() - start).seconds
            print(f'Processed {files_cnt} files in {seconds:.0f}s, ({seconds/files_cnt:.0f} sec/file)')
            Db.close()
    except KeyboardInterrupt:
        logger.info('metering is interrupted by user')
        if _GroupEquip.ready_to_analyze(event='the_end'):
            Analyzer.run('the_end')
    finally:
        Db.close()


if __name__ == '__main__':
    main()
