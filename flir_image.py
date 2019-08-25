import glob
import os
import datetime
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

import cv2 as cv
import numpy as np
from matplotlib import cm

import flir_image_extractor
from db import ImageFiles
from config import Config


class Cfg(Config):
    cache_thermal_allowed = True


class FlirImage:

    def __init__(self, fname_path, skip_thermal=False):
        assert os.path.exists(fname_path), f'file {fname_path} does not exists'
        self.fname_path = fname_path
        self.datetime = self.extract_datetime()
        self.fie = flir_image_extractor.FlirImageExtractor(image_suffix="v.jpg", thermal_suffix="t.png")
        cached_thermal = ImageFiles.get_cached_thermal(fname_path) if Cfg.cache_thermal_allowed else None
        self.fie.process_image(fname_path, cached_thermal, skip_thermal)
        self.flir_img = self.get_flir_image(fname_path)
        self.visual_img = self.fie.get_rgb_np()
        self.thermal_np = self.fie.get_thermal_np()
        thermal_normalized = (self.thermal_np - np.amin(self.thermal_np)) \
                             / (np.amax(self.thermal_np) - np.amin(self.thermal_np))
        self.thermal_img = np.array(np.uint8(cm.inferno(thermal_normalized) * 255))  # inferno,gray
        self.thermal_img = cv.cvtColor(self.thermal_img, cv.COLOR_RGBA2BGR)

        self.vis_therm_ratio = (self.visual_img.shape[0] / self.thermal_np.shape[0],
                                self.visual_img.shape[1] / self.thermal_np.shape[1])
        if self.flir_img.shape[0] > self.flir_img.shape[1]:  # vertical
            # transpose
            # self.flir_img = cv.transpose(self.flir_img)
            # self.visual_img = cv.transpose(self.visual_img)
            # self.thermal_np = cv.transpose(self.thermal_np)
            # self.thermal_img = cv.transpose(self.thermal_img)
            # rotate
            self.flir_img = np.rot90(self.flir_img)
            self.visual_img = np.rot90(self.visual_img)
            self.thermal_np = np.rot90(self.thermal_np)
            self.thermal_img = np.rot90(self.thermal_img)

        self.image_id = ImageFiles.save(self.datetime, self.flir_img,
                                        self.visual_img, self.thermal_np)
        self.skip_thermal = skip_thermal

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.fname_path}')"

    def extract_datetime(self):
        try:
            fname = os.path.basename(self.fname_path)[5:-4]  # remove path, 'flir_' and extension
            flir_datetime = datetime.datetime.strptime(fname, '%Y%m%dT%H%M%S')
        except ValueError:
            flir_datetime = None
        return flir_datetime

    def visual_xy_to_therm(self, visual_xy):
        if self.skip_thermal:
            return None
        # use ratio between visual and thermo
        # recalcualte visual_xy -> th_x, th_y
        return int(visual_xy[0] / self.vis_therm_ratio[0]), \
               int(visual_xy[1] / self.vis_therm_ratio[1])

    @staticmethod
    def get_flir_image(fname_path):
        image = cv.imread(fname_path)
        return image

    def point_temperature(self, therm_point_xy):
        return self.thermal_np[therm_point_xy[1], therm_point_xy[0]]  # x-->1, y-->0

    def save_visual_old(self, folder_to_save=None, prefix='', vis_suffix='_v.jpg'):
        path, fname = os.path.split(self.fname_path)
        if folder_to_save is None:
            folder_to_save = path
        else:
            os.makedirs(folder_to_save.rstrip('/'), exist_ok=True)
        fname_noext = fname[:-4]
        fname_path_out = f'{folder_to_save}/{prefix}{fname_noext}{vis_suffix}'
        cv.imwrite(fname_path_out, self.visual_img)
        print(f'visual image of {self.fname_path} saved to {fname_path_out}')


if __name__ == '__main__':

    def save_vis_image(fname_path, out_folder):
        # process_file(fname_path)
        fi = FlirImage(fname_path, skip_thermal=True)
        fi.save_visual_old(folder_to_save=out_folder)


    def vis_img_processing():
        total_time_sec = 0
        file_counter = 0
        root_fold = '../data/tests'
        # root_fold = '../data/calibr/att_3/lbl_3in_curved'
        subfold_list = ['rotate']  # os.listdir(root_fold):  # ['30', '60', '90','120']
        out_fold = f'{root_fold}/visual'
        for subfold in subfold_list:
            for fname_path in glob.glob(f'{root_fold}/{subfold}/*.jpg'):
                assert os.path.exists(fname_path), f'file {fname_path} not exist'
                start = datetime.datetime.now()

                save_vis_image(fname_path, f'{out_fold}/{subfold}')
                # split_video_to_vis_images(fname_path,f'{out_fold}/{subfold}')

                total_time_sec += (datetime.datetime.now() - start).seconds
                file_counter += 1
            print(f'files:{file_counter}   average time: {total_time_sec/file_counter} sec/file')


    vis_img_processing()
