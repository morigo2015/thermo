import os
import shutil
import glob
import math
from statistics import mean
import numpy as np
import cv2 as cv
import qr_read

import colors
from my_utils import Debug, KeyPoint, KeyPointList, MyMath
import config

inp_folder = f'../tmp'  # f'../data/calibr/att_3/lbl_3inch/visual/30' #
inp_mask = f'test90.jpg'
res_folder = f'../tmp/res_preproc'


class Cfg(config.Cfg):
    min_blob_diameter = 10  # minimal size (diameter) of blob
    max_relative_delta = 0.2  # default for approx comparison


class QrAnchorList:
    def __init__(self, image):
        self.image = image
        self.keypoints = self.get_blob_keypoints(image)
        self.qr_anchor_list = self.keypoints_2_qr_anchors()  # qr_anchor - set of 3 keypoints for 3 big qr_code quadrates

    def get_blob_keypoints(self, image):
        # input image --> list of blob keypoints
        params = cv.SimpleBlobDetector_Params()
        # Filter by Area.
        params.filterByArea = True
        params.minArea = MyMath.circle_area_by_diameter(Cfg.min_blob_diameter)

        # Set up the detector
        detector = cv.SimpleBlobDetector_create(params)
        # Detect blobs
        keypoints = detector.detect(image)
        blob_keypoints = KeyPointList(blob_detector_keypoints_list=keypoints)

        print('blob detector:', blob_keypoints)
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv.drawKeypoints(image, blob_keypoints, np.array([]), colors.BGR_BLUE,
                                             cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        Debug.log_image('im_with_keypoints')
        return blob_keypoints

    def is_in_list(self, kp1, kp2, kp3):
        return self.qr_anchor_list.count((kp1, kp2, kp3)) > 0

    def __repr__(self):
        item_lst = '\n\t' + '\n\t'.join([f'({item[0]}, {item[1]}, {item[2]})' for item in self.qr_anchor_list])
        # if not item_lst:
        #     item_lst = '<empty>'
        return f'{self.__class__.__name__}[{len(self.qr_anchor_list)}:{item_lst}'

    def __iter__(self):
        for elem in self.qr_anchor_list:
            yield elem

    def keypoints_2_qr_anchors(self):
        MyMath.set_max_relative_delta(Cfg.max_relative_delta)
        print('kp2anchors, before', self.keypoints)
        qr_anchors = []
        for kp in self.keypoints:
            for kp1 in self.keypoints:
                if kp1 is kp:
                    continue
                for kp2 in self.keypoints:
                    if kp2 is kp or kp2 is kp1:
                        continue
                    print('internal: ', kp, kp1, kp2)
                    if not (MyMath.approx_equal(kp.size, kp1.size) and MyMath.approx_equal(kp.size, kp2.size)):
                        continue
                    print('size check: ', kp, kp1, kp2)

                    # check if (kp1,kp,kp2) is right (90') triangle.
                    # conditions: |kp,kp1|~~|kp,kp2| and |kp,kp1|*sqrt(2)~~|kp1,kp2|
                    if not MyMath.approx_equal(kp.distance(kp1), kp.distance(kp2)):
                        continue
                    if not MyMath.approx_equal(kp.distance(kp1) * math.sqrt(2), kp1.distance(kp2)):
                        continue

                    print('append: ', kp, kp1, kp2)
                    if not self.is_in_list(kp, kp2, kp1):
                        qr_anchors.append((kp, kp1, kp2))

        print('anchors found:', qr_anchors)
        return qr_anchors


class QrArea:

    def __init__(self, image, qr_ancor):
        self.image = image
        self.qr_anchor = qr_ancor
        self.center = KeyPoint(size=0,
                               x=int(mean([kp.x for kp in self.qr_anchor])),
                               y=int(mean([kp.x for kp in self.qr_anchor])))
        self.radius = max([self.center.distance(kp) for kp in qr_ancor])

        self.qr_roi = self.image[self.center.x - self.radius:self.center.x + self.radius,
                      self.center.y - self.radius:self.center.y + self.radius]

        blur = cv.GaussianBlur(self.qr_roi, (5, 5), 0)
        ret, self.thresh_otsu = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        ret2, self.image_thresholded = cv.threshold(image, self.thresh_otsu, 255, cv.THRESH_BINARY)

        self.mask = np.zeros(self.image_thresholded.shape, dtype=np.uint8)
        self.mask[self.center.x - self.radius:self.center.x + self.radius,
        self.center.y - self.radius:self.center.y + self.radius] = 255
        Debug.log_image('mask')

        self.image_finished = self.image_thresholded
        self.image_thresholded[self.mask == 0] = 255
        Debug.log_image('image_finished')

    def __repr__(self):
        print(f'{self.__class__.__name__}({self.qr_anchor},c={self.center},r={self.radius}))')

    def draw_qr_area(self, image):
        if len(image.shape == 2):
            img_qr_area = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        else:
            img_qr_area = image.copy()
        for kp in self.qr_anchor:
            cv.circle(img_qr_area, (kp.x, kp.y), int(kp.size / 2), colors.BGR_GREEN, 1)
        cv.circle(img_qr_area, self.center, self.radius, colors.BGR_GREEN, 1)
        Debug.log_image('img_qr_area')
        return img_qr_area


def main():
    shutil.rmtree(res_folder, ignore_errors=True)
    os.makedirs(res_folder, exist_ok=True)
    tot_cnt, ok_cnt = 0, 0

    for fname_path in glob.glob(f'{inp_folder}/{inp_mask}'):
        Debug.set_log_image_names(fname_path, res_folder)
        image0 = cv.imread(fname_path, cv.IMREAD_GRAYSCALE)

        qr_anchor_list = QrAnchorList(image0) # qr_anchor - 3 big squares keypoints

        for qr_anchor in qr_anchor_list:

            qr_area = QrArea(image0, qr_anchor) # qr_are is a part of image0 with qr_code and all related methods
            # qr_area_img = qr_area.draw()
            # Debug.log_image('qr_area_img')

            qr_list = qr_read.QrMarkList(qr_area.image_finished)
            qr_list.draw_boxes()
            qr_list.draw_polygons()

            found = len(qr_list.qr_marks)

            ok_cnt += 1 if found else 0
        tot_cnt += 1
        print(f'{fname_path}: Total: {ok_cnt}/{tot_cnt} ({ok_cnt/tot_cnt*100:.2f}%)')


if __name__ == '__main__':
    main()

    # def find_otsu_threshold(self):
    #     # Otsu's thresholding after Gaussian filtering
    #     blur = cv.GaussianBlur(qr_anchor, (5, 5), 0)
    #     ret, th_otsu = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    #     Debug.log_image('th_otsu')
    #     ret2, image_thresholded = cv.threshold(image0, ret, 255, cv.THRESH_BINARY)
    #     Debug.log_image('image_thresholded')
    #     return image_thresholded
    #
    # def build_mask(self):
    #     # def do_mask(image_thresholded, x_c, y_c, r):
    #     self.mask = np.zeros(image_thresholded.shape, dtype=np.uint8)
    #     self.mask[y_c - r:y_c + r, x_c - r:x_c + r] = 255
    #     Debug.log_image('mask')
    #
    #     image_masked2 = image_thresholded.copy()
    #     image_masked2[mask == 0] = 255
    #     Debug.log_image('image_masked2')
    #     return image_masked2
    #

# def get_qr_area(image0, qr_anchor):
#     # kp3 = sorted(keypoints, key= lambda k: k.size, reverse=True)[0:3]
#     # x_c = int(sum([kp.pt[0] for kp in qr_anchor]) / 3)
#     # y_c = int(sum([kp.pt[1] for kp in qr_anchor]) / 3)
#     # r = max([math.hypot(x_c - kp.pt[0], y_c - kp.pt[1]) for kp in qr_anchor])
#     # r = int(r * 1.7)
#     qr_area = QrArea(image0, qr_anchor)
#     qr_area_img = qr_area.draw()
#     Debug.log_image('qr_area_img')
#
#     # qr_anchor=np.zeros((2*r,2*r),dtype=np.uint8)
#     qr_area = image0[y_c - r:y_c + r, x_c - r:x_c + r]
#     Debug.log_image('qr_area')
#     return r, x_c, y_c, qr_area


# def otsu_threshold(image0, qr_anchor):
#     # Otsu's thresholding after Gaussian filtering
#     blur = cv.GaussianBlur(qr_anchor, (5, 5), 0)
#     ret, th_otsu = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
#     Debug.log_image('th_otsu')
#     ret2, image_thresholded = cv.threshold(image0, ret, 255, cv.THRESH_BINARY)
#     Debug.log_image('image_thresholded')
#     return image_thresholded
#
#
# def do_mask(image_thresholded, x_c, y_c, r):
#     mask = np.zeros(image_thresholded.shape, dtype=np.uint8)
#     mask[y_c - r:y_c + r, x_c - r:x_c + r] = 255
#     Debug.log_image('mask')
#
#     image_masked2 = image_thresholded.copy()
#     image_masked2[mask == 0] = 255
#     Debug.log_image('image_masked2')
#     return image_masked2
#

#
# image_thresholded = otsu_threshold(image0, qr_area)
#
# image_masked2 = do_mask(image_thresholded, x_c, y_c, r)

# found = find_marks(qr_area.image_finished)
#
# def find_marks(image_masked2):
#     qr_list = qr_read.QrMarkList(image_masked2)
#     qr_list.draw_boxes()
#     qr_list.draw_polygons()
#     return len(qr_list.qr_marks)
#
