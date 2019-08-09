import os
import math
import sys
import inspect
import statistics
import datetime

import cv2 as cv
import numpy as np

import colors


class Debug:
    fname_path = None
    log_folder = None
    verbose = False  # 0/False - no debug, 1/True - debug, 2 - verbose debug

    @classmethod
    def set_log_image_names(cls, cur_fname_path, log_folder, verbose=False):  # пока оставил для qr_gen.py ***
        cls.fname_path = cur_fname_path
        cls.log_folder = log_folder
        cls.verbose = verbose

    @classmethod
    def set_params(cls, log_folder=None, input_file=None, verbose=None):
        if cls.verbose:
            Debug.print(f'Debug set params: '
                        f'log_folder={log_folder}, input={input_file}, verbose = {verbose}')
        if log_folder is not None:
            cls.log_folder = log_folder
        if input_file is not None:
            cls.fname_path = input_file
        if verbose is not None:
            cls.verbose = verbose

    @classmethod
    def log_image(cls, img_name, image=None):
        save_file_name = f'{cls.log_folder}/{os.path.basename(cls.fname_path)[:-4]}_{img_name}.jpg'
        if image is None:
            img = sys._getframe().f_back.f_locals[img_name]
        else:
            img = image
        # if cls.verbose >= 1:
        cv.imwrite(save_file_name, img)
        if cls.verbose > 1:
            print(f'{img_name} (shape={img.shape})saved to {save_file_name}')

    @classmethod
    def error(cls, message):
        print(f'ERROR!!! in func {inspect.stack()[1].function}: {message}')

    @classmethod
    def print(cls, message, verbose_level=None):
        if verbose_level is None:
            verbose_level = 1
        if verbose_level <= cls.verbose:
            print(message)


class KeyPoint:
    def __init__(self, x=None, y=None, size=0, xy=None, blob_detector_keypoint=None, ):
        if blob_detector_keypoint is not None:
            # super().__init__(keypoint['pt'][0],keypoint['pt'][1])
            self.x = int(blob_detector_keypoint.pt[0])
            self.y = int(blob_detector_keypoint.pt[1])
            self.size = int(blob_detector_keypoint.size)
        elif xy is not None:
            # super().__init__(xy[0],xy[1])
            self.x = xy[0]
            self.y = xy[1]
            self.size = size
        elif x is not None and y is not None:
            # super().__init__(x,y)
            self.x = x
            self.y = y
            self.size = size
        else:
            raise Exception(f'{self.__class__.__name__}.__init__('
                            f'keypoint={blob_detector_keypoint},size={size},xy={xy},x={x},y={y}) failed')

    def __repr__(self):
        return f'{self.__class__.__name__}({int(self.size)},({int(self.x)},{int(self.y)}))'

    def __str__(self):
        return f'({int(self.size)},({int(self.x)},{int(self.y)}))'

    def distance(self, kp2):
        return math.sqrt((kp2.x - self.x) ** 2 + (kp2.y - self.y) ** 2)

    def angle(self, kp1, kp2):
        # angle between (self,kp1) and (self,kp2), i.e. angle of (self) corner in triangle (kp1)(self)(kp2)
        return MyMath.angle(kp1, self, kp2)

    def draw(self, image, label, color=colors.BGR_GREEN):
        cv.circle(image, (self.x, self.y), 10, color, 2)
        if label is not None:
            cv.putText(image, label, (self.x, self.y), cv.FONT_HERSHEY_SIMPLEX, 2, color, 2)

    @classmethod
    def draw_list(cls, points_list, label_list, image):
        for kp, label in zip(points_list, label_list):
            kp.draw(image, label)

    def draw_beam(self, kp2, label, image, color=colors.BGR_CYAN):
        cv.line(image, (self.x, self.y), (kp2.x, kp2.y), color, 2)
        self.draw(image,label)

    @staticmethod
    def xy_to_offset(xy,anchor):
        # kp = KeyPoint(100, 100)
        # kp1 = KeyPoint(200, 100)   # x-axis
        # kp2 = KeyPoint(100, 200)
        # xy = (150,125)
        kp, kp1, kp2 = anchor
        ang = kp.angle(kp1,kp2)
        if ang > 0:  # swap kp1,2 to get kp2 pointed to x-axis in (kp1,kp,kp2) axes
            kp2,kp1 = kp1,kp2

        kp2_projx = KeyPoint(x=kp2.x,y=kp.y)
        ang2 = kp.angle(kp2_projx,kp2)  # angle between (kp,kp2) and image's x-axis

        mat = cv.getRotationMatrix2D(kp.xy(), ang2, 1.0)
        src = np.array([[[xy[0],xy[1]]]])
        res = cv.transform(src,mat)[0][0]   # rotate
        res = (res[0]-kp.x,res[1]-kp.y)     # shift
        res = (res[0]/kp.distance(kp2), res[1]/kp.distance(kp1))  # scale, kp2 - x-axis, kp1 - y-axis
        # print(res)
        return tuple(res)

    @staticmethod
    def offset_to_xy(offset,anchor):
        kp, kp1, kp2 = anchor
        ang = kp.angle(kp1,kp2)
        if ang > 0:
            kp2,kp1 = kp1,kp2
        offset = (offset[0]*kp.distance(kp2), offset[1]*kp.distance(kp1))  # scale
        offset = (offset[0]+kp.x, offset[1]+kp.y)  # shift

        kp2_projx = KeyPoint(x=kp2.x,y=kp.y)
        ang2 = kp.angle(kp2,kp2_projx)  # angle between (kp,kp2) and image's x-axis

        mat = cv.getRotationMatrix2D(kp.xy(), ang2, 1.0)
        src = np.array([[[offset[0],offset[1]]]])
        res = cv.transform(src,mat)[0][0]   # rotate
        return int(res[0]),int(res[1])


    # @staticmethod
    # def xy_to_offset(xy,anchor):
    #     # kp = KeyPoint(100, 100)
    #     # kp1 = KeyPoint(200, 100)   # x-axis
    #     # kp2 = KeyPoint(100, 200)
    #     # xy = (150,125)
    #     kp, kp1, kp2 = anchor
    #     ang = kp.angle(kp1,kp2)
    #     if ang > 0:
    #         kp2,kp1 = kp1,kp2
    #
    #     src_tri = np.array([ [kp.x,kp.y] for kp in [kp1,kp,kp2]]).astype(np.float32)
    #     dst_tri = np.array([ [0,100], [0,0], [100,0]]).astype(np.float32)
    #
    #     mat = cv.getAffineTransform(src_tri,dst_tri)
    #     src = np.array([[[xy[0],xy[1]]]])
    #     res = cv.transform(src,mat)
    #     # print(res)
    #     return res[0][0]

    # @staticmethod
    # def offset_to_xy(offset,anchor):
    #     # kp = KeyPoint(100, 100+100)
    #     # kp1 = KeyPoint(200, 100+100)   # x-axis
    #     # kp2 = KeyPoint(100, 200+100)
    #     # offset = (25,50)
    #     kp, kp1, kp2 = anchor
    #     ang = kp.angle(kp1,kp2)
    #     if ang > 0:
    #         kp2,kp1 = kp1,kp2
    #     src_tri = np.array([ [0,100], [0,0], [100,0]]).astype(np.float32)
    #     dst_tri = np.array([ [kp.x,kp.y] for kp in [kp1,kp,kp2]]).astype(np.float32)
    #     mat = cv.getAffineTransform(src_tri,dst_tri)
    #     src = np.array([[[offset[0],offset[1]]]])
    #     res = cv.transform(src,mat)
    #     # print(res)
    #     return tuple(res[0][0])

    def xy(self):
        return self.x, self.y

    def find_4th_corner(self, kp1, kp2):
        # find 4th corner for parallelogram: (kp1)(kp)(kp2)(kp4??), this corner is opposite to (kp) corner
        x = kp2.x + (kp1.x - self.x)
        y = kp2.y + (kp1.y - self.y)
        return KeyPoint(x=x, y=y)

    @staticmethod
    def expand(kp1, kp2, kp3, kp4, ratio: float):
        # expand square (kp1,kp2,kp3,kp4) in ratio times (image isn't resized, only corners moves out)
        keypoints = (kp1, kp2, kp3, kp4)
        c_x = statistics.mean((kp.x for kp in keypoints))
        c_y = statistics.mean((kp.y for kp in keypoints))
        return tuple(
            KeyPoint(x=int(kp.x + (kp.x - c_x) * ratio / 2), y=int(kp.y + (kp.y - c_y) * ratio / 2), size=kp.size)
            for kp in keypoints)

    @staticmethod
    def fit_to_shape(keypoints_tuple, shape):
        # return new tuple of keypoints where all x,y are in range (0:shape[0],0:shape[1]
        xmax = shape[0] - 1
        ymax = shape[1] - 1
        return tuple(KeyPoint(x=min(xmax, max(0, kp.x)), y=min(ymax, max(0, kp.y)), size=kp.size)
                     for kp in keypoints_tuple)

    @staticmethod
    def get_subimage(keypoints, image):
        # keypoints --> aligned (rotated to horizontal/vertical) sub image
        qr_area_contour = np.array([(kp.x, kp.y) for kp in keypoints], dtype=np.int32)

        # convert qr_area_corners to cv.Box2D: ( (center_x,center_y), (width,height), angle of rotation)
        rect = cv.minAreaRect(qr_area_contour)  # get Box2D for rotated rectangle

        # Get center, size, and angle from rect
        center, size, theta = rect
        # Convert to int
        center, size = tuple(map(int, center)), tuple(map(int, size))
        # Get rotation matrix for rectangle
        rot_matrix = cv.getRotationMatrix2D(center, theta, 1)
        # Perform rotation on src image
        dst = cv.warpAffine(image, rot_matrix, image.shape[:2])
        out = cv.getRectSubPix(dst, size, center)
        return out


class KeyPointList:

    def __init__(self, key_points_list=None, blob_detector_keypoints_list=None):
        if blob_detector_keypoints_list is not None:
            self.lst = [KeyPoint(blob_detector_keypoint=item) for item in blob_detector_keypoints_list]
        elif key_points_list is not None:
            self.lst = key_points_list
        else:
            raise Exception(f'{self.__class__.__name__}.__init__('
                            f'key_points_list={key_points_list},'
                            f'blob_detector_keypoints_list={blob_detector_keypoints_list}) failed')

    def __repr__(self):
        return f'{self.__class__.__name__}({",".join([str(item) for item in self.lst])})'

    def __str__(self):
        lst_str = "\n\t".join([str(item) for item in self.lst])
        return f'KeyPointsList[{len(self.lst)}]:\n\t{lst_str})'

    def __iter__(self):
        for elem in self.lst:
            yield elem


class MyMath:

    @staticmethod
    def approx_equal(v1, v2, tolerance):
        # check if v1 and v2 are approximately equal (i.e. relative difference <= max_relative_delta
        average = (v1 + v2) / 2
        delta = abs(v1 - v2)
        return delta / average <= tolerance

    @staticmethod
    def circle_area_by_diameter(diameter):
        return 3.14 * (diameter / 2) ** 2

    @staticmethod
    def angle(kp1: KeyPoint, kp2: KeyPoint, kp3: KeyPoint):
        # return andgle (in degrees) between [kp2,kp1] and [kp2,kp3]. I.e angle at kp2 in triangle (kp1,kp2,kp3)
        v0 = np.array([kp1.x, kp1.y]) - np.array([kp2.x, kp2.y])
        v1 = np.array([kp3.x, kp3.y]) - np.array([kp2.x, kp2.y])
        angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
        return np.degrees(angle)

class Misc:

    @staticmethod
    def dtime_to_str(dtime):
        return datetime.datetime.strftime(dtime, '%Y%m%dT%H%M%S')

    @staticmethod
    def str_to_dtime(dtime_str):
        return datetime.datetime.strptime(dtime_str, '%Y%m%dT%H%M%S')


if __name__ == '__main__':
    kp = KeyPoint(200,100)
    kp1 = KeyPoint(100,200)
    kp2 = KeyPoint(300,200)

    new_kp = KeyPoint(500,200)
    new_kp1 = KeyPoint(700,400)
    new_kp2 = KeyPoint(700,0)

    # xy = (130,105)
    for xy in [kp.xy(),kp1.xy(),kp2.xy(),(400,300),(200,300),(300,100),(100,100),(200,0),(300,300)]:
        new_xy = KeyPoint.xy_to_offset(xy,(kp,kp1,kp2))
        inv_xy = KeyPoint.offset_to_xy(new_xy,(new_kp,new_kp1,new_kp2))
        print(f'xy={xy[0]:.0f},{xy[1]:.0f}    '
              f'new={new_xy[0]:.0f},{new_xy[1]:.0f}   '
              f'inv={inv_xy[0]:.0f},{inv_xy[1]:.0f}')

