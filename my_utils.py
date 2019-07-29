import os
import shutil
import glob
import math
import sys
import numpy as np
import cv2 as cv
import qr_read

import colors

inp_folder = f'../tmp'  # f'../data/calibr/att_3/lbl_3inch/visual/30' #
inp_mask = f'test90.jpg'
res_folder = f'../tmp/res_preproc'


class Debug:
    fname_path = None
    res_folder = None
    verbose = None

    @classmethod
    def set_log_image_names(cls, cur_fname_path, res_folder, verbose=False):
        cls.fname_path = cur_fname_path
        cls.res_folder = res_folder
        cls.verbose = verbose

    @classmethod
    def log_image(cls, img_name):
        save_file_name = f'{cls.res_folder}/{os.path.basename(cls.fname_path)[:-4]}_{img_name}.jpg'
        img = sys._getframe().f_back.f_locals[img_name]
        cv.imwrite(save_file_name, img)
        if cls.verbose:
            print(f'{img_name} (shape={img.shape})saved to {save_file_name}')


class KeyPoint:
    def __init__(self, size=None, xy=None, x=None, y=None, blob_detector_keypoint=None, ):
        if blob_detector_keypoint is not None:
            # super().__init__(keypoint['pt'][0],keypoint['pt'][1])
            self.x = blob_detector_keypoint.pt[0]
            self.y = blob_detector_keypoint.pt[1]
            self.size = blob_detector_keypoint.size
        elif size is not None and xy is not None:
            # super().__init__(xy[0],xy[1])
            self.x = xy[0]
            self.y = xy[1]
            self.size = size
        elif size is not None and x is not None and y is not None:
            # super().__init__(x,y)
            self.x = x
            self.y = y
            self.size = size
        else:
            raise Exception(f'{self.__class__.__name__}.__init__('
                            f'keypoint={blob_detector_keypoint},size={size},xy={xy},x={x},y={y}) failed')

    def __repr__(self):
        return f'{self.__class__.__name__}({self.size},({self.x},{self.y}))'

    def __str__(self):
        return f'({int(self.size)},({int(self.x)},{int(self.y)}))'

    def distance(self,kp2):
        return math.sqrt((kp2.x-self.x)**2+(kp2.y-self.y)**2)

    @classmethod
    def test(cls):
        kp1 = KeyPoint(size=100, xy=(3, 4))
        kp2 = KeyPoint(size=200, x=8, y=9)
        print(f'kp1={kp1}, kp2={kp2}')
        kp5 = KeyPoint(blob_detector_keypoint={'size': 500, 'pt': (77, 88)})
        print(f'kp5={kp5}')
        # test incorrect init:
        # kp3 = KeyPoints(size=300,x=9)
        # kp4 = KeyPoints(size=400)


class KeyPointList:

    def __init__(self, key_points_list=None, blob_detector_keypoints_list=None):
        if blob_detector_keypoints_list is not None:
            self._lst = [KeyPoint(blob_detector_keypoint=item) for item in blob_detector_keypoints_list]
        elif key_points_list is not None:
            self._lst = key_points_list
        else:
            raise Exception(f'{self.__class__.__name__}.__init__('
                            f'key_points_list={key_points_list},blob_detector_keypoints_list={blob_detector_keypoints_list}) failed')

    def __repr__(self):
        return f'{self.__class__.__name__}({",".join([str(item) for item in self._lst])})'

    def __str__(self):
        lst_str = "\n\t".join([str(item) for item in self._lst])
        return f'KeyPointsList[{len(self._lst)}]:\n\t{lst_str})'

    def __iter__(self):
        for elem in self._lst:
            yield elem

    @staticmethod
    def test():
        kp1 = KeyPoint(size=100, xy=(5, 6))
        kp2 = KeyPoint(size=200, xy=(8, 9))
        print(f'kp1={kp1}, kp2={kp2}')
        kp_list = KeyPointList([kp1, kp2])
        print(f'kp_list={kp_list}')
        blob_list = [{'size': 1, 'pt': (11, 22)}]
        print(f'blob_list={KeyPointList(blob_detector_keypoints_list=blob_list)}')


class MyMath:
    max_relative_delta = 0.2 # default value

    @classmethod
    def set_max_relative_delta(cls,max_relative_delta):
        cls.max_relative_delta = max_relative_delta

    @classmethod
    def approx_equal(cls, v1, v2, max_relative_delta=None):
        # check if v1 and v2 are approximately equal (i.e. relative difference <= max_relative_delta
        if max_relative_delta is None:
            max_relative_delta = cls.max_relative_delta
        average = (v1 + v2) / 2
        delta = abs(v1 - v2)
        return delta / average <= max_relative_delta

    @staticmethod
    def circle_area_by_diameter(diameter):
        return 3.14 * (diameter / 2) ** 2


if __name__ == '__main__':
    # KeyPoint.test()
    KeyPointList.test()
