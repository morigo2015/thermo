import os
import math
import sys
import inspect
import statistics
import cv2 as cv
import numpy as np


# inp_folder = f'../tmp'  # f'../data/calibr/att_3/lbl_3inch/visual/30' #
# inp_mask = f'test90.jpg'
# log_folder = f'../tmp/res_preproc'
#

class Debug:
    fname_path = None
    log_folder = None
    verbose = False  # 0/False - no debug, 1/True - debug, 2 - verbose debug

    @classmethod
    def set_log_image_names(cls, cur_fname_path, log_folder, verbose=False):  # пока оставил для qr_gen.py ***
        cls.fname_path = cur_fname_path
        cls.res_folder = log_folder
        cls.verbose = verbose

    @classmethod
    def set_params(cls, log_folder=None, input_file=None, verbose=None):
        if log_folder is not None:
            cls.res_folder = log_folder
        if input_file is not None:
            cls.fname_path = input_file
        if verbose is not None:
            cls.verbose = verbose

    @classmethod
    def log_image(cls, img_name, image=None):
        save_file_name = f'{cls.res_folder}/{os.path.basename(cls.fname_path)[:-4]}_{img_name}.jpg'
        if image is None:
            img = sys._getframe().f_back.f_locals[img_name]
        else:
            img = image
        if cls.verbose >= 1:
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

    @staticmethod
    def test():
        kp1 = KeyPoint(size=100, xy=(5, 6))
        kp2 = KeyPoint(size=200, xy=(8, 9))
        print(f'kp1={kp1}, kp2={kp2}')
        kp_list = KeyPointList([kp1, kp2])
        print(f'kp_list={kp_list}')
        blob_list = [{'size': 1, 'pt': (11, 22)}]
        print(f'blob_list={KeyPointList(blob_detector_keypoints_list=blob_list)}')


#
# class Box:
#     # Box = collections.namedtuple('Box', 'left, top, width, height')
#     def __init__(self,left=None,top=None,width=None,height=None,sides=None):
#         if sides is None: # sides - tuple(left,top,width,height
#
#         self.left,self.top,self.width,self.height = left,top,width,height
#
#     def intersect(self,box):
#
#
#     def IoU(self,box):
#         # determine the (x, y)-coordinates of the intersection rectangle
#         xA = max(boxA[0], boxB[0])
#         yA = max(boxA[1], boxB[1])
#         xB = min(boxA[2], boxB[2])
#         yB = min(boxA[3], boxB[3])
#
#         # compute the area of intersection rectangle
#         interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
#
#         # compute the area of both the prediction and ground-truth
#         # rectangles
#         boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
#         boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
#
#         # compute the intersection over union by taking the intersection
#         # area and dividing it by the sum of prediction + ground-truth
#         # areas - the interesection area
#         iou = interArea / float(boxAArea + boxBArea - interArea)
#
#         # return the intersection over union value
#         return iou
#

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


if __name__ == '__main__':
    # KeyPoint.test()
    # KeyPointList.test()
    pass
    # print(MyMath.find_4th_corner(KeyPoint(0, 1), KeyPoint(1, 2), KeyPoint(2, 1)))
    # print(MyMath.find_4th_corner(KeyPoint(0, 1), KeyPoint(0, 0), KeyPoint(1, 0)))
    # kps = MyMath.stretch(KeyPoint(10, 10), KeyPoint(15, 10), KeyPoint(15, 15), KeyPoint(10, 15), 10)
    # print(kps)
    # kps_fit = MyMath.fit_to_shape(kps, (17, 17))
    # print(kps_fit)
