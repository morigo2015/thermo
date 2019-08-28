import numpy as np
import cv2 as cv
import os
import glob
from matplotlib import pyplot as plt

class Saver:
    def __init__(self, fname):
        self.fname_noext = fname[:-4]
        self.seqnum=0

    def save(self,img,suffix):
        img_fname = f'{self.fname_noext}_{self.seqnum}_{suffix}.jpg'
        cv.imwrite(img_fname,img)
        self.seqnum += 1

def proc(fname):
    fn = fname[:-4] # cut extension
    img = cv.imread(fname)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    cv.imwrite(fn+'_thresh.jpg',thresh)

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
    cv.imwrite(fn+'_open.jpg',opening)

    # sure background area
    sure_bg = cv.dilate(opening,kernel,iterations=3)
    cv.imwrite(fn+'_bg.jpg',sure_bg)

    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    cv.imwrite(fn+'_fg.jpg',sure_fg)

    unknown = cv.subtract(sure_bg,sure_fg)
    cv.imwrite(fn+'_unknown.jpg',unknown)

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv.watershed(img,markers)
    cv.imwrite(fn+'_imgpre.jpg',img)

    img[markers == -1] = [255,0,0]
    cv.imwrite(fn+'_imgfin.jpg',img)

def pre(img,sv):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    sv.log_image(gray, 'gray')

    blured = cv.medianBlur(gray,55)
    sv.log_image(blured, 'blured10')

    _, thresh = cv.threshold(blured,50,255,cv.THRESH_BINARY)
    sv.log_image(thresh, 'thresh')

    kernel = np.ones((5,5),np.uint8)
    open = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=10)
    sv.log_image(open, 'open')

    close = cv.morphologyEx(open, cv.MORPH_CLOSE, kernel, iterations=10)
    sv.log_image(close, 'close')

def analyze(img,sv):

    min_val = min(img.ravel())
    max_val = max(img.ravel())
    print(f'min={min_val} max={max_val}')
    # plt.hist(img.ravel(), 256, [0, 256])
    # plt.show()

    hist = cv.calcHist([img], [0], None, [20], [127, 256])
    plt.plot(hist)
    # plt.show()
    plt.savefig(sv.fname_noext+"_hist.jpg")

def show_hot(img,sv):

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sv.log_image(img, 'gray')

    blurred = cv.medianBlur(img,5)
    sv.log_image(blurred, 'blur')

    for thrsh_lvl in [50,100,150,200]:
        ret, thresh = cv.threshold(blurred, thrsh_lvl, 255, cv.THRESH_BINARY)
        # sv.save(thresh, 'thresh'+str(thrsh_lvl))

        kernel = np.ones((5, 5), np.uint8)
        open = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=5)
        # sv.save(open, 'open'+str(thrsh_lvl))

        close = cv.morphologyEx(open, cv.MORPH_CLOSE, kernel, iterations=5)
        sv.log_image(close, 'close' + str(thrsh_lvl))


def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv.Canny(image, lower, upper)

	# return the edged image
	return edged

def therm_segm(img,sv):
    # watershed segmentation. Hottest 20% are fg markers, coldest 20% - bg markers (environment)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.medianBlur(img,5)
    sv.log_image(blurred, 'blur')

    ret, hot_mask = cv.threshold(blurred, 100, 255, cv.THRESH_BINARY)
    sv.log_image(hot_mask, 'hot')

    ret, cold_mask = cv.threshold(blurred, 50, 255, cv.THRESH_BINARY_INV)
    sv.log_image(cold_mask, 'cold')

    for low_thresh in range(0,205,50):
        for high_thresh in range(low_thresh+50,255+5,50):
            sv.log_image(cv.Canny(blurred, low_thresh, high_thresh), f'canny_{low_thresh}_{high_thresh}')

for fname in glob.glob('../tmp/7/*_t.png'):
    print(f'fname = {fname}')
    sv = Saver(fname)
    img = cv.imread(fname)
    if img is None:
        print(f'can not open file {fname}')
        exit(1)
    sv.save(img,'start')

    # therm_segm(img,sv)
    # show_hot(img,sv)
    analyze(img,sv)
    # pre(img,sv)
    # proc(fname)
