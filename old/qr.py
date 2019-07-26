import requests
import glob
import os

import numpy as np
import cv2 as cv
import pyzbar.pyzbar as pyzbar

class QrCfg:
    # sheet params:
    vert_pixels = 2480
    horz_pixels = 3508
    # grid params:
    grid_cols = 2
    grid_rows = 1
    # single qr code params:
    size_pix = 150
    ecc_level:str = 'H'
    text_area_height = 50
    text_font = cv.FONT_HERSHEY_SIMPLEX
    text_font_size = 12
    text_font_thickness = 3
    sequence_start = 0

    @classmethod
    def show_str(cls):
        return "\n".join(
            [f'{v}={QrCfg.__dict__[v]}' for v in QrCfg.__dict__.keys()
             if not v.startswith('__') and not callable(getattr(cls,v))
             ]
            )

class QRgen:

    def __init__(self, size_pix =150, ecc_level:str ='H', text_area_height =50,
                 text_font=cv.FONT_HERSHEY_SIMPLEX, text_font_size =12, text_font_thickness =3,
                 sequence_start=0):
        self.size_pix = size_pix
        self.ecc_level = ecc_level
        self.text_area_height = text_area_height
        self.text_font_size = text_font_size
        self.text_font = text_font
        self.text_font_thickness = text_font_thickness
        self.last_seq_num = sequence_start # used for serializing

    def __gen_qr_img(self, data:str):
        # generate one image with QR
        size = str(self.size_pix)
        url = f"https://chart.googleapis.com/chart?chs={size}x{size}&cht=qr&chl={data}&chld={self.ecc_level}"
        response = requests.get(url)
        # print(f'url={url}')
        if response.status_code != 200:
            print(f'bad response while qr generating. URL={url}')
            return None
        image = np.asarray(bytearray(response.content), dtype="uint8")
        image = cv.imdecode(image, cv.IMREAD_COLOR)
        return image

    def __gen_txt_img(self,txt:str):
        textsize = cv.getTextSize(txt, self.text_font, 1, self.text_font_thickness)[0]
        txt_img = np.ones((self.text_area_height, self.size_pix, 3),dtype='uint8')
        txt_img *= 255
        # get coords based on boundary
        textX = int( (txt_img.shape[1] - textsize[0]) / 2)
        textY = int( (txt_img.shape[0] + textsize[1]) / 2)
        # add text centered on image
        cv.putText(txt_img, txt, (textX, textY), self.text_font, 1, (0,0,0), self.text_font_thickness)
        return txt_img

    def encode(self, data:str, txt:str =''):
        qr_img = self.__gen_qr_img(data)
        txt_img = self.__gen_txt_img(txt)
        img = np.concatenate((qr_img, txt_img), axis=0)
        return img

    def draw_qr_col_row(self, col, row):
        # get_image function for higher-levels classes (PageGrid)
        # enumerate cells by col, row
        return self.encode(f"{col}_{row}", f"col={col} row={row}")

    def draw_qr_serialized(self, col, row):
        # get_image function for higher-levels classes (PageGrid)
        # generate serialized qr (col,row is not used)
        self.last_seq_num += 1
        return self.encode(f"{self.last_seq_num}", f"{self.last_seq_num}")

    @staticmethod
    def test():
        qr = QRgen(size_pix=100, text_area_height=30, text_font_size=10, text_font_thickness=2)
        qr_img = qr.encode("tst1234", "tst1234")
        cv.imwrite("../tmp/test2.jpg", qr_img)


class PageGrid:

    def __init__(self, vert_pixels=2480, horz_pixels=3508, grid_cols=2, grid_rows=1):
        self.vert_pixels, self.horz_pixels = vert_pixels, horz_pixels
        self.img = np.ones((horz_pixels, vert_pixels, 3), dtype=np.uint8) * 255
        self.grid_cols, self.grid_rows = grid_cols, grid_rows
        self.cell_w = int(self.vert_pixels / self.grid_cols)
        self.cell_h = int(self.horz_pixels / self.grid_rows)

    def __cell_lu(self, c, r):
        # left-upper corner of the cell
        return (c * self.cell_w, r * self.cell_h)

    def __cell_rb(self, c, r):
        # right-bottom corner of the cell
        return ((c + 1) * self.cell_w - 1, (r + 1) * self.cell_h - 1)

    def add_cell_borders(self, color=(0, 0, 0), thickness=1):
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                cv.rectangle(self.img, self.__cell_lu(col, row), self.__cell_rb(col, row), color, thickness)
                # print(f'col={col} row={row} lu: {self.__cell_lu(col, row)} rb: {self.__cell_rb(col, row)}')

    def add_cell_images(self, get_image_func):
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                l, u = self.__cell_lu(col, row)
                r, b = self.__cell_rb(col, row)
                cell_image = get_image_func(col, row)
                ci_h, ci_w = cell_image.shape[0:2]  # cell image height, width (opposite to (x,y))
                # put cell_image in center of cell
                ci_u = int(u + (b - u) / 2 - ci_h / 2)  # cell image new upper
                ci_l = int(l + (r - l) / 2 - ci_w / 2)  # cell image new left
                print(f'col={col} row={row} ul: {u} {l} br: {b} {r} ci_w,h: {ci_w} {ci_h} ci_u,l: {ci_w} {ci_l}')
                self.img[ci_u:ci_u + ci_h, ci_l:ci_l + ci_w] = cell_image[0:ci_h, 0:ci_w]

    def save(self, file_name):
        cv.imwrite(file_name, self.img)

    @staticmethod
    def test():
        pg = PageGrid(grid_cols=4, grid_rows=11)
        pg.add_cell_borders()
        pg.add_cell_images(QRgen(size_pix=260, sequence_start=123456).draw_qr_col_row) # get_qr_serialized
        pg.save("../tmp/grid.jpg")


# ---------------------------------

class QrMark:

    def __init__(self, image):
        self.image = image
        img2 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        _, img2 = cv.threshold(img2, 127, 255, cv.THRESH_BINARY)
        self.marks = pyzbar.decode(img2)

    def draw_marks_boxes(self):
        for mark in self.marks:
            pt1 = (mark.rect.left, mark.rect.top)
            pt2 = (mark.rect.left + mark.rect.width, mark.rect.top + mark.rect.height)
            cv.rectangle(self.image, pt1, pt2, (0, 255, 0), cv.FILLED)

    def draw_marks_poly(self):
        pass

    def __str__(self):
        return f'QrMarkList( {[m.data+" " for m in self.marks]} )'

    @staticmethod
    def test():
        qr = QrMarks()
        ok_cnt = 0
        bad_cnt = 0
        fold = "../tmp/A/"
        for fn in glob.glob(f"{fold}/*.jpg"):
            img = cv.imread(fn)
            objs = qr.decode(img)
            if len(objs) == 0:  # no obj found
                res = '_NotFound'
                bad_cnt += 1
            else:
                ok_cnt += 1
                res = ''
                for obj in objs:
                    res += '_' + str(obj.data)
            print(f'{fn} = {res}')
            new_fn = f'{fold}/result/{os.path.basename(fn)[:-4]}{res}.jpg'
            cv.imwrite(new_fn, img)
        print(f'not found {bad_cnt} of {bad_cnt + ok_cnt}')


if __name__ == '__main__':
    # QRgen.test()
    # PageGrid.test()
    print (QrCfg.show())

