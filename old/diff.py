# image difference
import sys

import cv2
import numpy as np

def diff(target, input, blur):

    # convert to monochrome
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    input_gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)

    # blur
    if blur :
        target_gray = cv2.blur(target_gray,(blur,blur))
        input_gray = cv2.blur(input_gray,(blur,blur))

    # abs(subtract) pixel-wise
    diff_img = cv2.absdiff(target_gray,input_gray)

    # blend images
    blend_img = cv2.addWeighted(target_gray, 0.5, input_gray, 0.5, 0)

    # calculate metric
    s = np.sum(diff_img) / 255
    l = diff_img.size
    diff_res = s / l

    return diff_res, diff_img, blend_img

if __name__ == '__main__':
    img_folder = 'images'
    target_fname = sys.argv[1]
    input_fname = sys.argv[2]
    blur = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    target_img = cv2.imread(f'{img_folder}/{target_fname}.jpg', cv2.IMREAD_COLOR)
    input_img = cv2.imread(f'{img_folder}/{input_fname}.jpg', cv2.IMREAD_COLOR)

    # target_img = np.zeros(target_img.shape,np.uint8)
    # input_img = np.ones(target_img.shape,np.uint8) * 255

    diff_res, diff_img, blend_img = diff(target_img, input_img, blur)

    res_repr = f'{int(diff_res*10000):04d}'
    diff_fname = f'{img_folder}/d_{target_fname}_{input_fname}_{blur}_{res_repr}.jpg'
    blend_fname = f'{img_folder}/b_{target_fname}_{input_fname}_{blur}_{res_repr}.jpg'

    cv2.imwrite(diff_fname, diff_img)
    cv2.imwrite(blend_fname, blend_img)
    print(f'diff({target_fname},{input_fname},{blur}) = {diff_res}')