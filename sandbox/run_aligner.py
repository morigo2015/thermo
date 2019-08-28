from __future__ import print_function
import glob
import os
import shutil

import cv2
import numpy as np

MAX_MATCHES = 500
GOOD_MATCH_PERCENT = 0.15


def alignImages(im1, im2, matchFilename):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_MATCHES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite(matchFilename, imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h

def align_files(refFilename:str, imFilename:str, outFilename:str, blendedFilename:str, matchFilename:str):
    print("Reading reference image : ", refFilename)
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
    if imReference is None:
        print(f"Check refFilename: {refFilename}")
        exit(1)

    # Read image to be aligned
    print("Reading image to align : ", imFilename);
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
    if imReference is None:
        print(f"Check imFilename: {imFilename}")
        exit(1)

    print("Aligning images ...")
    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.

    print(f"shapes: im {im.shape}, imReference {imReference.shape}")
    imReg, h = alignImages(im, imReference, matchFilename)

    # Write aligned image to disk.
    print("Saving aligned image : ", outFilename);
    cv2.imwrite(outFilename, imReg)

    # Print estimated homography
    print("Estimated homography : \n", h)

    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_reg_gray = cv2.cvtColor(imReg, cv2.COLOR_BGR2GRAY)
    im_blended = cv2.addWeighted(im_gray, 0.5, im_reg_gray, 0.5, 0)
    im_blended = cv2.addWeighted(im, 0.5, imReg, 0.5, 0)
    print("Saving blended image : ", blendedFilename);
    cv2.imwrite(blendedFilename,im_blended)


if __name__ == '__main__':
    # Read reference image
    in_folder = '/media/sf_D_DRIVE/thermo/ubuntu-my-therm/data/result/'
    out_folder = '/media/sf_D_DRIVE/thermo/ubuntu-my-therm/tmp'
    path_to_imgs = 'ТестДом/Кабинет/КамераСобств'
    imgs_regex = '*_v.jpg' # regex include
    in_normalized_folder = "/media/sf_D_DRIVE/thermo/ubuntu-my-therm/data/Views_normal/"
    in_normalized_img = 'Камера_v.jpg'

    os.makedirs(os.path.join(out_folder, path_to_imgs), exist_ok=True)
    out_normalized_path_img = os.path.join(out_folder, path_to_imgs, "NORMAL_"+in_normalized_img)
    in_normalized_path_img = os.path.join(in_normalized_folder, path_to_imgs, in_normalized_img)
    shutil.copyfile(in_normalized_path_img, out_normalized_path_img)

    for in_fname_path in glob.glob(os.path.join(in_folder,path_to_imgs,imgs_regex)):
        out_f = os.path.basename(in_fname_path)
        out_f_ext = out_f[-4:]
        out_f_noext = out_f[:-4]
        out_f_path_noext = os.path.join(out_folder,path_to_imgs,out_f_noext)
        shutil.copyfile(in_fname_path, out_f_path_noext + "_orig.jpg")
        out_aligned_fname = out_f_path_noext + "_aligned.jpg"
        out_blended_fname = out_f_path_noext + "_blended.jpg"
        out_match_fname = out_f_path_noext + "_match.jpg"
        # print(in_normalized_img)
        # print('    ',in_fname_path)
        # print('        ',out_aligned_fname)
        # print('            ',out_blended_fname)
        # print('                ',out_match_fname)
        align_files(in_normalized_path_img,in_fname_path,out_aligned_fname,out_blended_fname,out_match_fname)


