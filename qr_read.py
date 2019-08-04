# qr read
# Pipeline:
# 1) image --> Blob list.
#       code: class Blob
#       Use OpenCV BlobDetector and additional conditions
# 2) Blob list --> Anchor list.
#       code: class QrAnchor
#       Anchor - 3 big square of qr code
#       Combine triplets of blobs and check conditions (size, angles, distances)
# 3) Anchor --> CandidatesAreas
#       code: class CandidateAreas
#       Prepare area around anchor for sending to qr detector (crop, affinewarp, thresholding)
# 4) CandidateAreas --> decoded qr code
#       code: class QrDecode
#       use pyzbar library wrapper

import os
import shutil
import glob

import numpy as np
import cv2 as cv
import pyzbar.pyzbar as pyzbar

import colors
from box import Box
from my_utils import Debug, KeyPoint, KeyPointList, MyMath
import config

# inp_folders = f'../tmp'  # f'../data/calibr/att_3/lbl_3inch/visual/30' #
inp_folders = f'../data/tests/visual/rot*'  #
# inp_fname_mask = f'flir_20190801T220319_v.jpg'
inp_fname_mask = f'*.jpg'

res_folder = f'../tmp/res_preproc'
verbose = 2

class Cfg(config.Cfg):
    min_blob_diameter = 10  # minimal size (diameter) of blob
    size_tolerance = 1  # default for approx comparison of sizes
    distance_tolerance = 1  # 0.3  # default for approx comparison of sizes
    angle_tolerance = 10  # how many degrees from 90' to treat angle as right(90')
    # dist between big square centers = 14; additional modules 2*4(border)+2*3.5(half of big square)+2(safety)=
    expand_ratio = (14 + 2 * 4 + 2 * 3.5 + 2) / 14
    use_otsu_threshold = False
    iou_threshold = 0.5  # minimal IoU to treat boxes are equal (for duplicate eliminating in qr_mark_decoded
    max_keypoints_number = 50  # exit if found more keypoints in image
    area_preparing_method = 'warp' # new, any else - old
    qr_module_size = 12  # module (small square in qr code) size
    qr_blank_width = 4  # white zone outside qr code (in modules)
    qr_side_with_blanks = 21 + 2 * qr_blank_width  # length of one side of qr code with white border (29) in modules


class Blob:
    def __init__(self, image):
        self.image = image
        self.blob_keypoints = self.get_blob_keypoints()

    def get_blob_keypoints(self):
        # input image --> list of blob keypoints
        params = cv.SimpleBlobDetector_Params()
        # Filter by Area.
        params.filterByArea = True
        params.minArea = MyMath.circle_area_by_diameter(Cfg.min_blob_diameter)

        # Set up the detector
        detector = cv.SimpleBlobDetector_create(params)
        # Detect blobs
        keypoints = detector.detect(self.image)
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        img_with_keypoints = cv.drawKeypoints(self.image, keypoints, np.array([]), colors.BGR_BLUE,
                                              cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        Debug.log_image('img_with_keypoints', img_with_keypoints)

        blob_keypoints = KeyPointList(blob_detector_keypoints_list=keypoints)
        Debug.print(f'keypoints found:{blob_keypoints}')
        return blob_keypoints


class QrAnchors:
    # anchor - triplet of big suqares (identifying pattern for qr-code area)
    # to find archor check all triplets of keypoints (kp1,kp,kp2):
    # 1) all sizes are similar
    # 2) |kp,kp1| ~~ |kp,kp2|
    # 3) angle of kp1,kp,kp2 close to 90'

    def __init__(self, blob_keypoints):
        self.keypoints = blob_keypoints.blob_keypoints
        self.image = blob_keypoints.image
        self.anchors = self.keypoints_2_qr_anchors()  # qr_anchor - set of 3 keypoints for 3 big qr_code quadrates
        self.draw_anchors()

    def keypoints_2_qr_anchors(self):
        # Debug.print(f'kp-->anchors, before:{self.keypoints}')
        Debug.print(f'hist (10,200,10):'
                    f'{np.histogram([i.size for i in self.keypoints.lst], bins=[i for i in range(10, 200, 10)])}',
                    verbose_level=2)
        if len(self.keypoints.lst) > Cfg.max_keypoints_number:
            Debug.error('Too many keypoints')
            exit(1)
        qr_anchors = []
        for kp in self.keypoints:
            for kp1 in self.keypoints:
                if kp1 is kp: continue
                for kp2 in self.keypoints:
                    if kp2 is kp or kp2 is kp1: continue
                    if not MyMath.approx_equal(kp.size, kp1.size, Cfg.size_tolerance): continue
                    if not MyMath.approx_equal(kp.size, kp2.size, Cfg.size_tolerance): continue
                    # check if (kp1,kp,kp2) is right (90') triangle
                    angle = kp.angle(kp1, kp2)
                    if not angle>0: continue  # not to have duplicates we throw away a lef-hand axes (or opposite :)
                    if not abs(angle - 90) < Cfg.angle_tolerance: continue
                    # check if |kp,kp1| ~~ |kp,kp2|
                    if not MyMath.approx_equal(kp.distance(kp1), kp.distance(kp2), Cfg.distance_tolerance): continue
                    # # skip (kp,kp1,kp2) if (kp,kp2,kp1) already exists
                    # if not qr_anchors.count((kp, kp2, kp1)) == 0: continue
                    Debug.print(f'append: {kp}, {kp1}, {kp2}')
                    qr_anchors.append((kp, kp1, kp2))
        Debug.print(f'found {len(qr_anchors)} anchors: {qr_anchors}')
        return qr_anchors

    def draw_anchors(self):
        for ind in range(len(self.anchors)):
            anchor = self.anchors[ind]
            img = cv.cvtColor(self.image, cv.COLOR_GRAY2BGR)
            for kp in anchor:
                cv.circle(img, (kp.x, kp.y), kp.size, colors.BGR_RED, 2)
            anchor[0].draw_beam(anchor[1],' ',img,colors.BGR_ORANGE)  # angle>0 ==> kp1 is xaxis (I suppose)
            Debug.log_image(f'anchor_{ind}', img)


class CandidateQrAreas:
    # CandidateQrArea - area prepared for sending to qr detector
    def __init__(self, anchors):
        # anchors --> candidate_areas:
        self.image = anchors.image
        self.anchors = anchors.anchors
        self.candidate_qr_areas = []
        for ind in range(len(self.anchors)):
            anchor = self.anchors[ind]
            kp, kp1, kp2 = anchor
            if Cfg.area_preparing_method== 'warp':  # todo remove, it's just a test
                # Extract subregion of qr_area from the entire image
                qr_area = self.get_subimage_warp(anchor, self.image)
                Debug.log_image('area_after_warp', qr_area)
            else: # old method: find 4th corner -> stretch -> fit_to_shape -> crop
                # find 4th point
                kp4 = kp.find_4th_corner(kp1, kp2)  # 3 squares --> 4th square
                # rectangle of 4 square centers --> qr_area rectangle
                corners = KeyPoint.expand(kp1, kp, kp2, kp4, Cfg.expand_ratio)
                # correct corners which are out of image (after stretching)
                qr_area_keypoints = KeyPoint.fit_to_shape(corners, self.image.shape)
                # Extract subregion of qr_area from the entire image
                qr_area = KeyPoint.get_subimage(qr_area_keypoints, self.image)
            # make otsu binarization if ordered in Cfg
            if Cfg.use_otsu_threshold:
                blur = cv.GaussianBlur(qr_area, (5, 5), 0)
                ret, candidate_area = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            else:
                candidate_area = qr_area
            # set axis info (cross_x,cross_y,xaxis_angle)
            Debug.log_image(f'finished_{ind}', candidate_area)
            self.candidate_qr_areas.append(candidate_area)

    @staticmethod
    def get_subimage_warp(anchor, image):
        pts_from = np.float32([(kp.x, kp.y) for kp in anchor])
        # set distance from the center of big square to the nearest edge of qr-code area
        square_center_dist = Cfg.qr_blank_width + 7 / 2
        pts_to = np.float32([[square_center_dist, square_center_dist],
                             [square_center_dist, Cfg.qr_side_with_blanks - square_center_dist],
                             [Cfg.qr_side_with_blanks - square_center_dist, square_center_dist]])
        pts_to *= Cfg.qr_module_size

        total_side_pix = Cfg.qr_side_with_blanks * Cfg.qr_module_size
        border_pix = Cfg.qr_blank_width * Cfg.qr_module_size

        mat = cv.getAffineTransform(pts_from, pts_to)
        qr_area = cv.warpAffine(image, mat, (total_side_pix, total_side_pix), None, cv.INTER_AREA,
                                cv.BORDER_CONSTANT, colors.BGR_WHITE)

        # color to fill cleaned blanks (to be similar to main image)
        border_color = qr_area[
            border_pix - Cfg.qr_module_size, border_pix - Cfg.qr_module_size]
        # make mask: 255 at border, 0 at qr code (between big squares)
        mask = np.ones((total_side_pix, total_side_pix), dtype=np.uint8)  # * 255
        mask[border_pix - Cfg.qr_module_size * 2:total_side_pix - border_pix + Cfg.qr_module_size * 2,
             border_pix - Cfg.qr_module_size * 2:total_side_pix - border_pix + Cfg.qr_module_size * 2] = 0
        # apply mask to qr_area (to clean background if qr code is curved)
        qr_area[mask == 1] = border_color
        return qr_area


class QrMark:
    # extended item from pyzabar.decode() list

    def __init__(self, pyzbar_obj,area,anchor):
        self.code = pyzbar_obj.data
        self.area = area
        self.anchor = anchor
        self.rect = pyzbar_obj.rect
        self.box = Box(corners=(self.rect.left, self.rect.top, self.rect.width, self.rect.height))

    def draw_box(self, image, color=colors.BGR_GREEN):
        cv.rectangle(image,
                     (self.box.startX, self.box.startY), (self.box.endX, self.box.endY), color, cv.FILLED)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.code},{self.box}'  # ,{self.polygon})'


class QrDecode:
    def __init__(self, candidate_areas):
        self.image = candidate_areas.image
        self.areas = candidate_areas.candidate_qr_areas
        self.anchors = candidate_areas.anchors

        # areas --> qr_decoded_marks:
        marks = []
        for ind in range(len(self.areas)):
            area = self.areas[ind]
            pyzbar_objs = pyzbar.decode(area)
            if not len(pyzbar_objs):
                continue

            # ToDo решить что делать если вдруг несколько кодов в кадре (так не должно быть)
            if len(pyzbar_objs) > 1:
                Debug.error(f'Multiple codes ({len(pyzbar_objs)}) found in {area}')

            mark = QrMark(pyzbar_objs[0],self.areas[ind],self.anchors[ind])
            marks.append(mark)
            Debug.log_image(f'found_{ind}_{mark.code}', area)

        # remove duplicates from marks
        # duplicates rarely created if several triplets looks like anchor while referencing to the same qr code area
        self.qr_decoded_marks = []
        for inp_m in marks:
            if all([not inp_m.code == out_m.code for out_m in
                    self.qr_decoded_marks]):  # no items in final list are near
                self.qr_decoded_marks.append(inp_m)
        Debug.print(f'decoded marks list after removing duplicates:{self.qr_decoded_marks}]')


def process_file(fname_path):
    image0 = cv.imread(fname_path, cv.IMREAD_GRAYSCALE)
    Debug.log_image('image0')

    blobs = Blob(image0)
    anchors = QrAnchors(blobs)
    candidate_areas = CandidateQrAreas(anchors)
    decoded_qrs = QrDecode(candidate_areas)

#####
    if not decoded_qrs.qr_decoded_marks:
        return 0
    qr_mark = decoded_qrs.qr_decoded_marks[0]
    anchor = qr_mark.anchor
    if fname_path.endswith('041_v.jpg'):
        xy = (1112,508)
        cv.circle(image0,xy,10,colors.BGR_YELLOW,2)
        cv.imwrite('../tmp/offset_target.jpg',image0)
        offset = KeyPoint.xy_to_offset(xy,anchor)
        print(f'offset={offset}')
    offset = (36,-198)
    new_xy = tuple(KeyPoint.offset_to_xy(offset,anchor))
    img = cv.cvtColor(image0,cv.COLOR_GRAY2BGR)
    cv.circle(img, new_xy, 10, colors.BGR_RED, 4)
    Debug.log_image(f'offset_{os.path.basename(fname_path)[:-4]}.jpg', img)

    ####

    return len(decoded_qrs.qr_decoded_marks)  # cnt marks found


def main():
    shutil.rmtree(res_folder, ignore_errors=True)
    os.makedirs(res_folder, exist_ok=True)
    Debug.set_params(log_folder=res_folder, verbose=verbose)

    for folder in sorted(glob.glob(f'{inp_folders}')):
        if not os.path.isdir(folder):
            continue

        files_cnt = 0
        files_found = 0
        for fname_path in glob.glob(f'{folder}/{inp_fname_mask}'):
            Debug.set_params(input_file=fname_path)

            ok_cnt = process_file(fname_path)

            if ok_cnt:
                files_found += 1
            files_cnt += 1
            print(
                f'\t\t{fname_path}: {ok_cnt} marks.  Files: '
                f'{files_found}/{files_cnt} rate={100*files_found/files_cnt:.2f}%')

        print(f'Folder {folder}: {files_found}/{files_cnt} rate={100*files_found/files_cnt:.2f}%')


if __name__ == '__main__':

    main()

