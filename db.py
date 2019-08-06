import os

import cv2 as cv
import numpy as np

from config import Config
from my_utils import Misc


class Cfg(Config):
    inp_folders = f'../data/tests/rot*'  #
    inp_fname_mask = f'*013.jpg'
    out_folder = f'../tmp/out'
    log_folder = f'../tmp/res_preproc'
    verbose = 2
    images_subfolder = 'images'  # images subfolder in {out_folder}


class Db:
    # api to database operation

    @staticmethod
    def save_reading_to_db(dtime, meter_id, image_id, temperature):
        print('save reading to db: ', dtime, meter_id, temperature, image_id)

    @staticmethod
    def get_meters_from_db(qr_code):
        print('get meters from db: ',qr_code)
        stub_meter = (1, (36, -198))  # meter_id, (offset_x,offset_y)
        return [stub_meter]

    @staticmethod
    def save_images_to_db(dtime_str, flir_img_fname, visual_img_fname, thermal_np_fname):
        print('save images to db: ', dtime_str, flir_img_fname, visual_img_fname, thermal_np_fname)
        image_id = 0
        return image_id

    @staticmethod
    def load_images_from_db(image_id):
        print('load images from db: ', image_id)
        return '', '', ''


class ImageFiles:

    @staticmethod
    def save(dtime, flir_img, visual_img, thermal_np):
        folder = str(dtime.date())
        folder_path = f'{Cfg.images_subfolder}/{folder}'
        os.makedirs(Cfg.out_folder + '/' + folder_path, exist_ok=True)
        dtime_str = Misc.dtime_to_str(dtime)
        flir_img_fname = f'{folder_path}/{dtime_str}_f.jpg'
        visual_img_fname = f'{folder_path}/{dtime_str}_v.jpg'
        thermal_np_fname = f'{folder_path}/{dtime_str}_t'
        cv.imwrite(f'{Cfg.out_folder}/{flir_img_fname}', flir_img)
        cv.imwrite(f'{Cfg.out_folder}/{visual_img_fname}', visual_img)
        np.save(f'{Cfg.out_folder}/{thermal_np_fname}', thermal_np)
        image_id = Db.save_images_to_db(dtime_str,
                                        flir_img_fname, visual_img_fname, thermal_np_fname)
        print(f'save images {folder_path}: {dtime_str}')
        return image_id

    @staticmethod
    def load(image_id):
        flir_img_fname, visual_img_fname, thermal_np_fname = Db.load_images_from_db(image_id)
        flir_img = cv.imread(f'{Cfg.out_folder}/{flir_img_fname}')
        visual_img = cv.imread(f'{Cfg.out_folder}/{visual_img_fname}')
        thermal_np = np.load(f'{Cfg.out_folder}/{thermal_np_fname}')
        return flir_img, visual_img, thermal_np
