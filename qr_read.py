import os
import shutil
import glob
import math
from statistics import mean
import collections

import numpy as np
import cv2 as cv
import pyzbar.pyzbar as pyzbar

import colors
from my_utils import Debug, KeyPoint, KeyPointList, MyMath
import config

inp_folder = f'../tmp'  # f'../data/calibr/att_3/lbl_3inch/visual/30' #
inp_mask = f'test90.jpg'
res_folder = f'../tmp/res_preproc'
Box = collections.namedtuple('Box', 'left, top, width, height')
Point = collections.namedtuple('Point', 'x, y')


class Cfg(config.Cfg):
    min_blob_diameter = 10  # minimal size (diameter) of blob
    max_relative_delta = 0.2  # default for approx comparison
    radius_multiplyer = 1.7  # increase radius of qr_acnhor for qr_area ****


class QrMark:

    def __init__(self, pyzbar_obj):
        self.code = pyzbar_obj.data
        rect = pyzbar_obj.rect
        self.box = Box(rect.left, rect.top, rect.width, rect.height)
        self.polygon = [Point(p.x, p.y) for p in pyzbar_obj.polygon]  # pyzbar.polygon = List[Points(x,y)]

    def center(self):
        return Point(self.box.left + int(self.box.width / 2), self.box.top + int(self.box.height / 2))

    def draw_box(self, image, color=colors.BGR_GREEN):
        cv.rectangle(image,
                     (self.box.left, self.box.top), (self.box.left + self.box.width, self.box.top + self.box.height),
                     color, cv.FILLED)

    def draw_polygon(self, image, color=colors.BGR_BLUE):
        pts = np.array([list(p) for p in self.polygon], np.int32)
        pts = pts.reshape((-1, 1, 2))  # just from polylines tutorial
        cv.polylines(image, [pts], True, color, 3)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.code},{self.box},{self.polygon})'


class QrMarkList:

    def __init__(self, image):
        # create list of QrMark for all qr codes in image
        self.image = image
        # img2 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        img2 = image  # *****
        # _, img2 = cv.threshold(img2, 127, 255, cv.THRESH_BINARY)
        self.qr_marks = [QrMark(pyzbar_obj) for pyzbar_obj in pyzbar.decode(img2)]

    def draw_boxes(self, color=colors.BGR_GREEN):
        for mark in self.qr_marks:
            mark.draw_box(self.image, color)

    def draw_polygons(self, color=colors.BGR_BLUE):
        for mark in self.qr_marks:
            mark.draw_polygon(self.image, color)

    def __repr__(self):
        return f'{self.__class__.__name__}[{",".join([str(m.code) for m in self.qr_marks])}]'


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
        im_with_keypoints = cv.drawKeypoints(image, keypoints, np.array([]), colors.BGR_BLUE,
                                             cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        Debug.log_image('im_with_keypoints')
        return blob_keypoints

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
                    # print('internal: ', kp, kp1, kp2)
                    if not (MyMath.approx_equal(kp.size, kp1.size) and MyMath.approx_equal(kp.size, kp2.size)):
                        continue
                    # print('size check: ', kp, kp1, kp2)

                    # check if (kp1,kp,kp2) is right (90') triangle.
                    # conditions: |kp,kp1|~~|kp,kp2| and |kp,kp1|*sqrt(2)~~|kp1,kp2|
                    if not MyMath.approx_equal(kp.distance(kp1), kp.distance(kp2)):
                        continue
                    if not MyMath.approx_equal(kp.distance(kp1) * math.sqrt(2), kp1.distance(kp2)):
                        continue

                    if not qr_anchors.count((kp, kp2, kp1)) > 0:
                        print('append: ', kp, kp1, kp2)
                        qr_anchors.append((kp, kp1, kp2))

        print(f'found {len(qr_anchors)} anchors:', qr_anchors)
        return qr_anchors


class QrArea:

    def __init__(self, image, qr_anchor):
        self.image = image
        self.qr_anchor = qr_anchor
        self.center = KeyPoint(size=0,
                               x=int(mean([kp.x for kp in self.qr_anchor])),
                               y=int(mean([kp.y for kp in self.qr_anchor])))
        self.radius = int(max([self.center.distance(kp) for kp in qr_anchor]))
        self.radius = int(self.radius * Cfg.radius_multiplyer)
        print('QrArea init;', self)
        self.draw_qr_area()

        image_copy = self.image.copy()
        image_copy = cv.rectangle(image_copy, (self.center.x - self.radius, self.center.y - self.radius),
                                  (self.center.x + self.radius, self.center.y + self.radius), colors.BGR_RED, 1)
        Debug.log_image('image_before_roi', image_copy)
        self.qr_roi = self.image[self.center.x - self.radius:self.center.x + self.radius,
                      self.center.y - self.radius:self.center.y + self.radius]
        Debug.log_image('qr_roi', self.qr_roi)

        blur = cv.GaussianBlur(self.qr_roi, (5, 5), 0)
        ret, self.thresh_otsu = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        Debug.log_image('thresh_otsu', self.thresh_otsu)

        ret2, self.image_thresholded = cv.threshold(image, ret, 255, cv.THRESH_BINARY)
        Debug.log_image('image_thresholded', self.image_thresholded)

        self.mask = np.zeros(self.image_thresholded.shape, dtype=np.uint8)
        self.mask[self.center.x - self.radius:self.center.x + self.radius,
        self.center.y - self.radius:self.center.y + self.radius] = 255
        Debug.log_image('mask', self.mask)

        self.image_finished = self.image_thresholded
        self.image_thresholded[self.mask == 0] = 255
        Debug.log_image('image_finished', self.image_finished)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.qr_anchor},c=({self.center.x},{self.center.y}),r={self.radius}))'

    def draw_qr_area(self):
        if len(self.image.shape) == 2:
            img_qr_area = cv.cvtColor(self.image, cv.COLOR_GRAY2BGR)
        else:
            img_qr_area = self.image.copy()
        for kp in self.qr_anchor:
            cv.circle(img_qr_area, (kp.x, kp.y), int(kp.size / 2), colors.BGR_GREEN, 1)
        cv.circle(img_qr_area, (self.center.x, self.center.y), self.radius, colors.BGR_GREEN, 1)
        Debug.log_image('img_qr_area')
        return img_qr_area


def main():
    shutil.rmtree(res_folder, ignore_errors=True)
    os.makedirs(res_folder, exist_ok=True)
    tot_cnt, ok_cnt = 0, 0

    for fname_path in glob.glob(f'{inp_folder}/{inp_mask}'):
        Debug.set_log_image_names(fname_path, res_folder, True)
        image0 = cv.imread(fname_path, cv.IMREAD_GRAYSCALE)
        Debug.log_image('image0')

        qr_anchor_list = QrAnchorList(image0)  # qr_anchor - 3 big squares keypoints

        for qr_anchor in qr_anchor_list:
            print('qr anchor::', qr_anchor[0], qr_anchor[1], qr_anchor[2])
            qr_area = QrArea(image0, qr_anchor)  # qr_are is a part of image0 with qr_code and all related methods
            # qr_area_img = qr_area.draw()
            # Debug.log_image('qr_area_img')

            qr_list = QrMarkList(qr_area.image_finished)
            qr_list.draw_boxes()
            qr_list.draw_polygons()

            found = len(qr_list.qr_marks)

            ok_cnt += 1 if found else 0
        tot_cnt += 1
        print(f'{fname_path}: Total: {ok_cnt}/{tot_cnt} ({ok_cnt/tot_cnt*100:.2f}%)')


if __name__ == '__main__':

    def test_scale():
        folder = '../tmp/calibr_result/90'
        image0 = cv.imread(f'{folder}/*.jpg')
        # image0 = cv.cvtColor(image0,cv.COLOR_BGR2GRAY)
        assert image0.shape[0] > 0
        # for scale in np.arange(0.1,2,0.1):
        for iter in range(1, 6):
            # image = cv.resize(image0,None,fx=scale,fy=scale,interpolation=cv.INTER_AREA)
            image0 = cv.resize(image0, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
            # _, image0 = cv.threshold(image0,127,255,cv.THRESH_BINARY)
            image = np.copy(image0)
            scale = iter
            cv.imwrite(f'{folder}/tt0_{scale}.png', image)
            qr_list = QrMarkList(image)
            qr_list.draw_boxes()
            qr_list.draw_polygons()
            cv.imwrite(f'{folder}/tt_{scale}.png', image)
            print(f'{scale}: {len(qr_list.qr_marks)} marks found')
        # print(qr_list)


    def test():
        inp_root = f'../data/calibr/att_3/lbl_3inch/visual'
        out_root = f'../tmp/res'
        for fold in ['30', '60', '90', '120']:  # ,'150','180']:
            print(f'==== {fold} ======')
            inp_folder = f'{inp_root}/{fold}'
            out_folder = f'{out_root}/{fold}'
            os.makedirs(out_folder, exist_ok=True)
            ok, tot = 0, 0
            for fn in glob.glob(f'{inp_folder}/*.jpg'):
                image0 = cv.imread(fn)
                file_ok = 0
                max_mark = 0
                best_scales = []
                for scale in [1.0, 0.5, 2.0, 0.25, 4.0]:
                    image = cv.resize(image0, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
                    qr_list = QrMarkList(image)
                    qr_list.draw_boxes()
                    qr_list.draw_polygons()
                    found = len(qr_list.qr_marks)
                    file_ok += 1 if found else 0
                    if found == max_mark:
                        best_scales.append(scale)
                    if found > max_mark:
                        max_mark = found
                        best_scales = [scale]
                    fname = f'{out_folder}/{found}_{scale}_{os.path.basename(fn)[:-4]}.png'
                    cv.imwrite(fname, image)
                    # print(f'{fname}: {found} marks.')
                if file_ok:
                    ok += 1
                tot += 1
                print(f'Total: {ok}/{tot}.      {fn}: {max_mark} marks found. Best scale = {best_scales}')


    # test_scale()
    # test()
    main()
