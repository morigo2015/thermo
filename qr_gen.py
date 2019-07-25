import requests
import datetime
import os
import json
import collections
import numpy as np
import cv2 as cv
import qrcode # external lib for qr generation (alternative to using chart.google.com)

import config

class QrCfg(config.Cfg):

    debug_level = 0 # 0 - no debug, 1 - grid only, 2 - all

    # sheet params (landscape):
    sheet_width_mm = 287 # 287
    sheet_height_mm = 201 # 201
    left_adjustment_mm = 4
    top_adjustment_mm = 2
    printer_dpi = 300
    horz_pixels = (sheet_width_mm) * int(printer_dpi/25.4)
    vert_pixels = (sheet_height_mm) * int(printer_dpi/25.4)
    left_adjust_pixels = left_adjustment_mm * int(printer_dpi/25.4)
    top_adjust_pixels = top_adjustment_mm * int(printer_dpi/25.4)

    # grid params:
    grid_cols = 11
    grid_rows = 4
    image_rotation = True # counter clockwise 90'
    grid_border = True
    border_color = (0,0,0)
    border_thickness = 1

    # single qr code params:
    qr_image_generator = 'qrcode' # 'google'
    size_pix = 'auto' # calculate as max to fit in grid cell
    ecc_level:str = 'H'
    text_area_height = 50
    text_font = cv.FONT_HERSHEY_SIMPLEX
    text_font_size = 32
    text_font_thickness = 3
    sequence_start = 1000000
    qr_white_margin = 10
    img_width_modules = 21+2*qr_white_margin # if ecc_level='H' then img is 21*21 modules + 2 * border(=4)
    qr_border = True # make border for single qr code

class QRgen:

    def __init__(self):
        self.last_seq_num=QrCfg.sequence_start

    @staticmethod
    def even_size_pix(size):
        return int(size/QrCfg.img_width_modules)*QrCfg.img_width_modules # multiple 21+2*4

    def _gen_qr_by_qrcode(self, data:str):
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=int(QrCfg.size_pix/QrCfg.img_width_modules),  # 21 - modules of version 1+ borders (2*4)
            border=QrCfg.qr_white_margin,
        )
        qr.add_data(data)
        qr.make() # fit=False)
        pil_img = qr.make_image(fill_color="black", back_color="white")
        # pil_img.save('../tmp/test_pil.png','png')
        img0 = np.array(pil_img).astype(np.uint8)*255
        # cv.imwrite('../tmp/test_cv0.png',img0)
        cv_img = cv.cvtColor( img0,cv.COLOR_GRAY2BGR) #,img2)
        # cv.imwrite('../tmp/test_cv.png',cv_img)
        return cv_img

    def _gen_qr_by_google(self, data:str):
        size = str(QrCfg.size_pix)
        ecc_level = 'H' #  we always need maximum ecc level (30%)
        url = f"https://chart.googleapis.com/chart?chs={size}x{size}&cht=qr&chl={data}&chld={ecc_level}"
        response = requests.get(url)
        if QrCfg.debug_level>1:
            print(f'url={url}')
        if response.status_code != 200:
            print(f'bad response while qr generating. URL={url}')
            return None
        image = np.asarray(bytearray(response.content), dtype="uint8")
        image = cv.imdecode(image, cv.IMREAD_COLOR)
        if QrCfg.qr_border:
            cv.rectangle(image,(0,0),(image.shape[0]-1,image.shape[1]-1),(0,0,0),1)
        return image

    def _gen_qr_img(self, data:str):
        # generate one image with QR
        if QrCfg.qr_image_generator == 'google':
            return self._gen_qr_by_google(data)
        elif QrCfg.qr_image_generator == 'qrcode':
            return self._gen_qr_by_qrcode(data)
        else:
            print(f'illegal qr generator {QrCfg.qr_image_generator}')
            exit(1)

    def _gen_txt_img(self, txt:str):
        textsize = cv.getTextSize(txt, QrCfg.text_font, 1, QrCfg.text_font_thickness)[0]
        txt_img = np.ones((QrCfg.text_area_height, QrCfg.size_pix, 3),dtype='uint8')
        txt_img *= 255
        # get coords based on boundary
        textX = int( (txt_img.shape[1] - textsize[0]) / 2)
        textY = int( (txt_img.shape[0] + textsize[1]) / 2)
        # add text centered on image
        cv.putText(txt_img, txt, (textX, textY), QrCfg.text_font, 1, (0,0,0), QrCfg.text_font_thickness)
        return txt_img

    def encode(self, data:str, txt:str =''):
        qr_img = self._gen_qr_img(data)
        txt_img = self._gen_txt_img(txt)
        img = np.concatenate((qr_img, txt_img), axis=0)
        return img

    def draw_qr_col_row(self, row, col):
        # get_image function for higher-levels classes (PageGrid)
        # enumerate cells by col, row
        return self.encode(f"{col}_{row}", f"col={col} row={row}")

    def draw_qr_serialized(self, row, col):
        # get_image function for higher-levels classes (PageGrid)
        # generate serialized qr (col,row is not used)
        self.last_seq_num += 1
        return self.encode(f"{self.last_seq_num}", f"{self.last_seq_num}")


class PageGrid:

    def __init__(self, get_image_func =QRgen().draw_qr_serialized):
        self.img = np.ones((QrCfg.horz_pixels, QrCfg.vert_pixels, 3), dtype=np.uint8) * 255
        self.cell_w = int(QrCfg.horz_pixels / QrCfg.grid_cols)
        self.cell_h = int(QrCfg.vert_pixels / QrCfg.grid_rows)
        QrCfg.size_pix = QRgen.even_size_pix( self.calculate_size_pix() )

        self._add_cell_images(get_image_func)

        if QrCfg.grid_border:
            self._add_cell_borders()
        self.add_adjustments()

    def calculate_size_pix(self):
        img_space_w = int(QrCfg.horz_pixels / QrCfg.grid_cols)
        img_space_w -= QrCfg.image_rotation if QrCfg.text_area_height else 0
        img_space_h = int(QrCfg.vert_pixels / QrCfg.grid_rows)
        img_space_h -= QrCfg.image_rotation if 0 else QrCfg.text_area_height
        return min(img_space_w, img_space_h) - 1


    def _cell_lu(self, r, c):
        # left-upper corner of the cell
        return (c * self.cell_w, r * self.cell_h)

    def _cell_rb(self, r, c):
        # right-bottom corner of the cell
        return ((c + 1) * self.cell_w - 1, (r + 1) * self.cell_h - 1)

    def _add_cell_borders(self):
        for row in range(QrCfg.grid_rows):
            for col in range(QrCfg.grid_cols):
                l, u = self._cell_lu(row, col)
                r, b = self._cell_rb(row, col)
                cv.rectangle(self.img, (u,l), (b,r),  # don't know why it works transponded only (u,l) instead of (l,u)
                             QrCfg.border_color, QrCfg.border_thickness)
                if QrCfg.debug_level:
                    print(f'col={col} row={row} lu: {l},{u} rb: {r},{b}')

    def _add_cell_images(self, get_image_func):
        for row in range(QrCfg.grid_rows):
            for col in range(QrCfg.grid_cols):
                l, u = self._cell_lu(row, col)
                r, b = self._cell_rb(row, col)
                cell_image = get_image_func(row, col)
                if QrCfg.image_rotation:
                    cell_image = cv.rotate(cell_image,cv.ROTATE_90_COUNTERCLOCKWISE)
                ci_w, ci_h = cell_image.shape[0:2]  # cell image height, width (opposite to (x,y))
                # put cell_image in center of cell
                ci_u = int(u + (b - u) / 2 - ci_h / 2)  # cell image new upper
                ci_l = int(l + (r - l) / 2 - ci_w / 2)  # cell image new left
                if QrCfg.debug_level:
                    print(f'col={col} row={row} ul: {u} {l} br: {b} {r} ci_w,h: {ci_w} {ci_h} ci_u,l: {ci_w} {ci_l}')
                assert ci_h <= (b-u), f"Image height is {ci_h} while grid cell height is {b-u} only"
                assert ci_w <= (r-l), f"Image width is {ci_w} while grid cell width is {r-l} only"
                self.img[ci_l:ci_l + ci_w, ci_u:ci_u + ci_h] = cell_image[0:ci_w, 0:ci_h]

    def add_adjustments(self):
        # left adjust
        left_adjust = np.ones((QrCfg.left_adjust_pixels, self.img.shape[1], 3), dtype=np.uint8) * 255
        cv.rectangle(left_adjust,(0,0),(left_adjust.shape[1],left_adjust.shape[0]),(222,222,222),-1)
        cv.line(left_adjust,(0,0),(left_adjust.shape[1], 0),(0,0,0),thickness=2)
        self.img = np.concatenate((self.img,left_adjust), axis=0)
        # top adjust
        top_adjust = np.ones((self.img.shape[0], QrCfg.top_adjust_pixels, 3), dtype=np.uint8) * 255
        cv.rectangle(top_adjust,(0,0),(top_adjust.shape[1],top_adjust.shape[0]),(222,222,222),-1)
        cv.line(top_adjust,(0,0),(0, top_adjust.shape[0]),(0,0,0),thickness=2)
        self.img = np.concatenate((top_adjust,self.img), axis=1)
        if QrCfg.debug_level:
            print(f'adjustments added: left={QrCfg.left_adjust_pixels} pix, top={QrCfg.top_adjust_pixels} pix')

    def save(self, fname_path):
        cv.imwrite(fname_path, self.img)
        print(f'QR grid saved to {fname_path}')


if __name__ == '__main__':

    def qrgen_test():
        qr = QRgen()
        qr_img = qr.encode("tst1234", "tst1234")
        cv.imwrite("../tmp/test2.jpg", qr_img)

    def generate_grid():
        start = datetime.datetime.now()
        # generate grid
        pg = PageGrid()
        pg.save("../tmp/grid.jpg")

        if QrCfg.debug_level:
            print(f'Total time = {(datetime.datetime.now()-start).seconds} sec')
            print('Config:\n', QrCfg)

    # QrCfg.size_pix=2000
    # qrgen_test()

    QrCfg.grid_cols = 2
    QrCfg.grid_rows = 2
    # QrCfg.debug = 2
    # # QrCfg.load_json('../configs/qr_grid_11_4.json')
    generate_grid()
    # # QrCfg.save_json('../configs/qr_grid_11_4.json')

