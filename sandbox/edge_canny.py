# import the necessary packages
import numpy as np
import argparse
import glob
import cv2
import os

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)

	# return the edged image
	return edged

    # image_file = "../tmp/inp1_t.png"
    # load the image, convert it to grayscale, and blur it slightly
for image_file in glob.glob("../tmp/4/*"):
    print(image_file)
    image = cv2.imread(image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # apply Canny edge detection using a wide threshold, tight
    # threshold, and automatically determined threshold
    wide = cv2.Canny(blurred, 10, 200)
    tight = cv2.Canny(blurred, 225, 250)
    auto = auto_canny(blurred)

    # show the images
    # cv2.imshow("Original", image)
    # cv2.imshow("Edges", np.hstack([wide, tight, auto]))

    fname = image_file[:-4]
    cv2.imwrite(f"{fname}_wide.jpg", wide)
    cv2.imwrite(f"{fname}_tight.jpg",tight)
    cv2.imwrite(f"{fname}_auto.jpg", auto)
    # cv2.waitKey(0)