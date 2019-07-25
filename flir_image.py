import glob
import os
import datetime

import numpy as np
import cv2 as cv

import qr_read
import flir_image_extractor


class FlirImage:

    def __init__(self, fname_path):
        assert os.path.exists(fname_path),f'file {fname_path} does not exists'
        self.fname_path = fname_path
        self.datetime = self.extract_datetime(fname_path)
        self.fie = flir_image_extractor.FlirImageExtractor(image_suffix="v.jpg", thermal_suffix="t.png")
        self.fie.process_image(fname_path)
        self.visual_np = self.fie.get_rgb_np()
        self.thermal_np = self.fie.get_thermal_np()
        self.marks_list = qr_read.QrMarkList(self.visual_np)
        # self.meters = [Meter(mark,self.thermal_np) for mark in self.marks_list.marks] # init meters:
        self.vis_therm_ratio = ( self.visual_np.shape[0]/self.thermal_np.shape[0],
                                 self.visual_np.shape[1]/self.thermal_np.shape[1])

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.fname_path}')"

    def extract_datetime(self, fname_path):
        try:
            fname = os.path.basename(fname_path)[:-4] # remove path and extension
            flir_datetime = datetime.datetime.strptime(fname,'flir_%Y%m%dT%H%M%S')
        except ValueError:
            flir_datetime = None
        return flir_datetime

    def visual_xy_to_therm(self, visual_xy):
        # use ratio between visual and thermo
        # recalcualte visual_xy -> th_x, th_y
        return ( int(visual_xy[0]/self.vis_therm_ratio[0]), int(visual_xy[1]/self.vis_therm_ratio[1]) )

    def point_temperature(self, therm_point_xy):
        return self.thermal_np[therm_point_xy[0], therm_point_xy[1]] # **** check it ****

    def get_mark_temper(self, mark):
        mark_vis_center = mark.center()
        mark_therm_center = self.visual_xy_to_therm(mark_vis_center)
        temperature = self.point_temperature(mark_therm_center)
        return temperature

    def save_visual(self, file_path):
        pass


if __name__ == '__main__':

    def process_file(fname_path):
        fi = FlirImage(fname_path)
        dt=fi.datetime
        if not fi.marks_list.qr_marks:
            print( f'can not find mark in {fname_path}')
            return
        # assert fi.marks_list.qr_marks,'marks not found'
        mark = fi.marks_list.qr_marks[0]
        print(fname_path, mark.code, fi.get_mark_temper(mark) )

    total_time_sec = 0
    file_counter = 0

    for fname_path in glob.glob('../tmp/flir3/flir*.jpg'):
        assert os.path.exists(fname_path),f'file {fname_path} not exist'
        start = datetime.datetime.now()

        process_file(fname_path)

        total_time_sec += (datetime.datetime.now() - start).seconds
        file_counter += 1
    print(f'files:{file_counter}   average time: {total_time_sec/file_counter} sec/file')

