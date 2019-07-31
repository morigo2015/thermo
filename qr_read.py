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
    size_tolerance = 1  # default for approx comparison of sizes
    distance_tolerance = 0.3  # default for approx comparison of sizes
    angle_tolerance = 10  # how many degrees from 90' to treat angle as right(90')
    # dist between big square centers = 14; additional modules 2*4(border)+2*3.5(half of big square)+2(safety)=
    stretch_ratio = (14 + 2 * 4 + 2 * 3.5 + 2) / 14
    use_otsu_threshold = False


class QrAnchorList:
    def __init__(self, image):
        self.image = image
        self.keypoints = self.get_blob_keypoints(image)
        self.qr_anchor_list = self.keypoints_2_qr_anchors()  # qr_anchor - set of 3 keypoints for 3 big qr_code quadrates
        self.draw_all_anchors()

    @staticmethod
    def draw_anchor(image, anchor, label):
        img = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        for kp in anchor:
            cv.circle(img, (kp.x, kp.y), kp.size, colors.BGR_RED, 1)
        Debug.log_image(label, img)

    def draw_all_anchors(self):
        for i in range(len(self.qr_anchor_list)):
            self.draw_anchor(self.image,self.qr_anchor_list[i],f'anchor_{i}')

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
        print('kp2anchors, before', self.keypoints)
        qr_anchors = []
        for kp in self.keypoints:
            for kp1 in self.keypoints:
                if kp1 is kp:
                    continue
                for kp2 in self.keypoints:
                    if kp2 is kp or kp2 is kp1:
                        continue
                    if not MyMath.approx_equal(kp.size, kp1.size, Cfg.size_tolerance):
                        continue
                    if not MyMath.approx_equal(kp.size, kp2.size, Cfg.size_tolerance):
                        continue

                    # check if (kp1,kp,kp2) is right (90') triangle.
                    if not abs(abs(kp.angle(kp1, kp2)) - 90) < Cfg.angle_tolerance:
                        continue

                    # check if |kp,kp1| ~~ |kp,kp2|
                    if not MyMath.approx_equal(kp.distance(kp1), kp.distance(kp2), Cfg.distance_tolerance):
                        continue

                    if not qr_anchors.count((kp, kp2, kp1)) == 0:
                        continue

                    print('append: ', kp, kp1, kp2)
                    qr_anchors.append((kp, kp1, kp2))

        print(f'found {len(qr_anchors)} anchors:', qr_anchors)
        return qr_anchors


class QrArea:

    def __init__(self, image, qr_anchor):
        self.image = image
        self.qr_anchor = qr_anchor

        # find qr_area corners:
        kp, kp1, kp2 = qr_anchor
        kp4 = kp.find_4th_corner(kp1, kp2)  # 3 squares --> 4th square
        qr_area = KeyPoint.stretch(kp1, kp, kp2, kp4, Cfg.stretch_ratio)
        self.qr_area_keypoints = KeyPoint.fit_to_shape(qr_area, self.image.shape)

        # Extract subregion
        self.qr_area = KeyPoint.getSubImage(self.qr_area_keypoints, self.image)
        Debug.log_image('qr_area_aligned', self.qr_area)

        if Cfg.use_otsu_threshold:
            blur = cv.GaussianBlur(self.qr_area, (5, 5), 0)
            ret, self.finished = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        else:
            self.finished = self.qr_area # just to test
        Debug.log_image('thresh_finished', self.finished)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.qr_anchor},c=({self.center.x},{self.center.y}),r={self.radius}))'

    def draw_qr_area(self,image0,label):
        image = cv.cvtColor(image0,cv.COLOR_GRAY2BGR)
        for kp in self.qr_anchor:
            cv.circle(image, (kp.x, kp.y), int(kp.size / 2), colors.BGR_GREEN, 1)

        # cv.circle(img_qr_area, (self.center.x, self.center.y), self.radius, colors.BGR_GREEN, 1)
        qr_area_contour = np.array([(kp.x, kp.y) for kp in self.qr_area_keypoints], dtype=np.int32)

        # convert qr_area_corners to cv.Box2D: ( (center_x,center_y), (width,height), angle of rotation)
        rect = cv.minAreaRect(qr_area_contour)  # get Box2D for rotated rectangle
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(image, [box], 0, colors.BGR_RED, 2)
        Debug.log_image(label,image)


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

    def draw_boxes(self, image, color=colors.BGR_GREEN):
        for mark in self.qr_marks:
            mark.draw_box(image, color)

    def draw_polygons(self, image, color=colors.BGR_BLUE):
        for mark in self.qr_marks:
            mark.draw_polygon(image, color)

    def __repr__(self):
        return f'{self.__class__.__name__}[{",".join([str(m.code) for m in self.qr_marks])}]'


def main():
    shutil.rmtree(res_folder, ignore_errors=True)
    os.makedirs(res_folder, exist_ok=True)
    tot_cnt, ok_cnt = 0, 0

    for fname_path in glob.glob(f'{inp_folder}/{inp_mask}'):
        Debug.set_log_image_names(fname_path, res_folder, True)
        image0 = cv.imread(fname_path, cv.IMREAD_GRAYSCALE)
        Debug.log_image('image0')

        qr_anchor_list = QrAnchorList(image0).qr_anchor_list  # qr_anchor - 3 big squares keypoints

        for anchor_num in range(len(qr_anchor_list)):
            qr_anchor = qr_anchor_list[anchor_num]
            print(f'qr anchor[{anchor_num}]:', qr_anchor[0], qr_anchor[1], qr_anchor[2])
            qr_area = QrArea(image0, qr_anchor)  # qr_are is a part of image0 with qr_code and all related methods

            qr_list = QrMarkList(qr_area.finished)

            if len(qr_list.qr_marks):
                ok_cnt += 1
                print(f'\t\t\tfound_{anchor_num}_{qr_list.qr_marks[0].code}')
                qr_area.draw_qr_area(image0,f'found_{anchor_num}_{qr_list.qr_marks[0].code}')
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
