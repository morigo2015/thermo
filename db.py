import os
import sqlite3
import logging
import collections
import cv2 as cv
import numpy as np

from config import Config
from my_utils import Misc

logger = logging.getLogger('thermo.' + 'db')


class Cfg(Config):
    inp_folders = f'../data/tests/rot*'  #
    # inp_fname_mask = f'*013.jpg'
    out_folder = f'../tmp/out'
    log_folder = f'../tmp/res_preproc'
    verbose = 2
    db_path = f'../sql/thermo_debug.db'
    images_subfolder = 'images'  # images subfolder in {out_folder}


class Db:
    # views/tables structures:
    ReadingsShortRecord = \
        collections.namedtuple('ReadingsShortRecord',
                               'dtime, meter_id, temperature')
    ReadingsRecord = \
        collections.namedtuple('ReadingsRecord',
                               'dtime, dtime_sec, meter_id, image_id, temperature')
    ReadingsHistRecord = \
        collections.namedtuple('ReadingsHistRecord',
                               'dtime, dtime_sec, meter_id, image_id, temperature, min_atmo, avg_atmo, max_atmo')
    OneValueRecord = collections.namedtuple('OneValueRecord', 'value')
    MeterEquipRecord = collections.namedtuple('MeterEquipRecord', 'meter_id, atmo_flg, equip_id')

    # api to database operation

    conn = None
    cur = None

    @classmethod
    def connect(cls):
        cls.conn = sqlite3.connect(Cfg.db_path)
        cls.conn.row_factory = sqlite3.Row
        cls.cur = cls.conn.cursor()

    @classmethod
    def close(cls):
        if cls.conn is None:
            return
        # cls.conn.commit()
        cls.conn.close()
        cls.cur = None
        cls.conn = None

    @classmethod
    def check_conn(cls):
        if cls.conn is None:
            cls.connect()

    @classmethod
    def select(cls, query, query_arg, record_ntuple, empty_ok=True):
        cls.exec(query, query_arg)
        result = cls.cur.fetchall()
        if not len(result) and empty_ok is False:
            logger.error(f'Error:: records for {query}({query_arg}) !!!')
            return None
        res_tup_list = [tuple(r) for r in result]
        res = [record_ntuple(*r) for r in res_tup_list]
        return res

    @classmethod
    def insert_one(cls, table_name, record_ntuple):
        query = f'insert into {table_name} values ({", ".join(["? "]*len(record_ntuple))})'
        query_args = tuple(record_ntuple)
        logger.debug(f'insert query={query} args={query_args}')
        cls.exec(query, query_args)

    @classmethod
    def exec(cls, query, query_args=None):
        cls.check_conn()
        if query_args is None:
            query_args = ()
        cnt = cls.cur.execute(query, query_args).rowcount
        cls.conn.commit()
        return cnt

    @classmethod
    def get_meters_from_db(cls, qr_code):
        # stub_meter = (1, (36, -198))  # meter_id, (offset_x,offset_y)
        cls.check_conn()
        cls.cur.execute('select * from Marks where code=?', (qr_code.decode('utf-8'),))
        result = cls.cur.fetchall()
        if not len(result):
            logger.error(f'Error:: Unknown code {qr_code} !!!')
            return None
        mark_id = int(result[0]["mark_id"])

        cls.cur.execute('select * from Meters where mark_id=?', (mark_id,))
        result = cls.cur.fetchall()
        if not len(result):
            logger.error(f'Error:: No meters in db for mark_id = {mark_id} (code {qr_code}) !!!')
            return None
        meters = [(rec["meter_id"], rec["offset_x"], rec["offset_y"]) for rec in result]
        logger.info(f'Get meters from db: code={qr_code},  mark_id={mark_id}, meters={meters}')
        return meters

    @classmethod
    def save_reading_to_db(cls, dtime, meter_id, image_id, temperature):
        logger.info(f'save reading to db: {dtime}, {meter_id}, {image_id}, {temperature}')
        cls.check_conn()
        cls.cur.execute('insert into Readings values (?, strftime("%s",?), ?, ?, ?)',
                        (Misc.dtime_to_str(dtime), Misc.dtime_to_sqlite_str(dtime),
                         meter_id, image_id, temperature))
        cls.conn.commit()

    @classmethod
    def save_images_to_db(cls, dtime_str, flir_img_fname, visual_img_fname, thermal_np_fname):
        logger.info(f'save images to db: {dtime_str}, {flir_img_fname}, {visual_img_fname}, {thermal_np_fname}')
        cls.check_conn()
        cls.cur.execute('insert into Images '
                        '(datetime, flir_fname, vis_fname, therm_fname)'
                        'values (?, ?, ?, ?)',
                        (dtime_str, flir_img_fname, visual_img_fname, thermal_np_fname))
        image_id = cls.cur.lastrowid  # sqlite specific!!!! return rowid as PK field value
        cls.conn.commit()
        return image_id

    @classmethod
    def load_images_from_db(cls, image_id):
        # stub
        print('load images from db: ', image_id)
        cls.check_conn()
        row = cls.cur.execute('select * '
                              'from Images where image_id=?', (image_id,)).fetchone()
        cls.conn.commit()

        return row['datetime'], row['flir_fname'], row['vis_fname'], row['therm_fname']

    _meter_equip = None

    @classmethod
    def meter_to_equip(cls, meter_id):
        if cls._meter_equip is None:
            meter_equip_records = cls.select('select * from meter_equip', (), cls.MeterEquipRecord)
            if not len(meter_equip_records):
                logger.error(f'empty view _meter_equip!')
                return -1
            cls._meter_equip = dict([(r.meter_id, r.equip_id) for r in meter_equip_records])
        try:
            equip_id = cls._meter_equip[meter_id]
        except KeyError:
            logger.error(f'Unknown meter_id={meter_id}')
            equip_id = -1
        return equip_id


class ImageFiles:

    @staticmethod
    def save(dtime, flir_img, visual_img, thermal_np):
        folder = str(dtime.date())
        folder_path = f'{Cfg.images_subfolder}/{folder}'
        os.makedirs(Cfg.out_folder + '/' + folder_path, exist_ok=True)
        dtime_str = Misc.dtime_to_str(dtime)
        flir_img_fname = f'{folder_path}/{dtime_str}_f.jpg'
        visual_img_fname = f'{folder_path}/{dtime_str}_v.jpg'
        thermal_np_fname = f'{folder_path}/{dtime_str}_t.npy'
        cv.imwrite(f'{Cfg.out_folder}/{flir_img_fname}', flir_img)
        cv.imwrite(f'{Cfg.out_folder}/{visual_img_fname}', visual_img)
        np.save(f'{Cfg.out_folder}/{thermal_np_fname}', thermal_np)
        image_id = Db.save_images_to_db(dtime_str,
                                        flir_img_fname, visual_img_fname, thermal_np_fname)
        logger.info(f'save images {folder_path}: {dtime_str}, image_id = {image_id}')
        return image_id

    @staticmethod
    def load(image_id):
        # not tested yet
        dtime_str, flir_img_fname, visual_img_fname, thermal_np_fname = \
            Db.load_images_from_db(image_id)
        flir_img = cv.imread(f'{Cfg.out_folder}/{flir_img_fname}')
        visual_img = cv.imread(f'{Cfg.out_folder}/{visual_img_fname}')
        thermal_np = np.load(f'{Cfg.out_folder}/{thermal_np_fname}', allow_pickle=True)
        return dtime_str, flir_img, visual_img, thermal_np

    @staticmethod
    def get_cached_thermal(flir_fname_path):
        # flir_fname_path --> date_str
        dtime_str = os.path.basename(flir_fname_path)[5:-4]  # cut 'flir_' and '.jpg'
        date_str = str(Misc.str_to_dtime(dtime_str).date())  # yyymmddThhmmss -> yyyy-mm-dd
        thermal_fname_path = f'{Cfg.out_folder}/{Cfg.images_subfolder}/{date_str}/{dtime_str}_t.npy'
        if not os.path.exists(thermal_fname_path):
            return None
        else:
            thermal_np = np.load(thermal_fname_path, allow_pickle=True)
            return thermal_np


import collections

if __name__ == '__main__':

    # Db.connect()
    ReadingShortRecord = collections.namedtuple('ReadingShortRecord', 'dtime, meter_id, temperature')
    readings = Db.select('select * from readings_short order by datetime', (), ReadingShortRecord)
    print(len(readings))
    for r in readings:
        print(tuple(r))

    # Db.close()
