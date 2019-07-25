import glob
import os
import collections

import numpy as np
import cv2 as cv
import pyzbar.pyzbar as pyzbar

import colors

Box = collections.namedtuple('Box','left, top, width, height')
Point = collections.namedtuple('Point', 'x, y')

class QrMark:

    def __init__(self, pyzbar_obj):
        self.code = pyzbar_obj.data
        rect=pyzbar_obj.rect
        self.box = Box(rect.left,rect.top,rect.width,rect.height)
        self.polygon = [ Point(p.x,p.y) for p in pyzbar_obj.polygon] # pyzbar.polygon = List[Points(x,y)]

    def center(self):
        return Point(self.box.left+int(self.box.width/2), self.box.top+int(self.box.height/2))

    def draw_box(self,image,color=colors.BGR_GREEN):
        cv.rectangle(image,
                     (self.box.left,self.box.top), (self.box.left+self.box.width,self.box.top+self.box.height),
                     color, cv.FILLED)

    def draw_polygon(self,image,color=colors.BGR_BLUE):
        pts = np.array([list(p) for p in self.polygon], np.int32)
        pts = pts.reshape((-1,1,2)) # just from polylines tutorial
        cv.polylines(image, [pts], True, color,3)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.code},{self.box},{self.polygon})'


class QrMarkList:

    def __init__(self, image):
        # create list of QrMark for all qr codes in image
        self.image = image
        img2 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        _, img2 = cv.threshold(img2, 127, 255, cv.THRESH_BINARY)
        self.qr_marks = [QrMark(pyzbar_obj) for pyzbar_obj in pyzbar.decode(img2)]

    def draw_boxes(self,color=colors.BGR_GREEN):
        for mark in self.qr_marks:
            mark.draw_box(self.image,color)

    def draw_polygons(self,color=colors.BGR_BLUE):
        for mark in self.qr_marks:
            mark.draw_polygon(self.image,color)

    def __repr__(self):
        return f'{self.__class__.__name__}[{",".join([str(m.code) for m in self.qr_marks])}]'


if __name__ == '__main__':
    image = cv.imread('../tmp/grid.jpg')
    assert image.shape[0]>0
    qr_list = QrMarkList(image)
    qr_list.draw_boxes()
    qr_list.draw_polygons()
    print(qr_list.qr_marks[0])
    print(qr_list)
    cv.imwrite('../tmp/21-draw.jpg',image)

