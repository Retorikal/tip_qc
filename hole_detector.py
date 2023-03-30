import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import ipywidgets as widgets


def read_image(path, ratio=0.5):
    img = cv.imread(path, flags=cv.IMREAD_GRAYSCALE)
    return resize(img, ratio=ratio)


def read_image_color(path, ratio=0.5):
    img = cv.imread(path)
    return resize(img, ratio=ratio)


def resize(img, ratio):
    width = int(img.shape[1] * ratio)
    height = int(img.shape[0] * ratio)
    dim = (width, height)
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)


def blur(img, size=41):
    return cv.GaussianBlur(img, (size, size), 0)


def circ_kernel(size):
    return cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size))


def remove_small_contours(img, minsize):
    contours, _ = cv.findContours(
        img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[-2:]

    small_contours = []
    for contour in contours:
        if cv.contourArea(contour) < minsize:
            small_contours.append(contour)

    cv.drawContours(img, small_contours, contourIdx=-1,
                     color=0, thickness=-1)

    return img


def kernel(size):
    return np.ones((size, size), np.float32)


def crop_image(img, center, area_size):
    reach = int(area_size / 2)
    return img[
        center[1] - reach:center[1] + reach,
        center[0] - reach:center[0] + reach]


def circle_mask(diameter):
    radius = int(diameter / 2)
    center = (radius, radius)
    mask = np.zeros((diameter, diameter), dtype=np.uint8)
    cv.circle(mask, center, radius, 255, thickness=-1)
    return mask


def apply_mask(img, mask):
    return np.bitwise_and(img, mask)


def locate_interest_contours(img, stray_area_thresh_ratio=0.1):
    # Accepts preprocessed image and gets the hole's centroid location
    contours, hierarchy = cv.findContours(
        img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[-2:]

    if hierarchy is None:
        return None, None, None

    hierarchy = hierarchy[0]
    areas = [cv.contourArea(c) for c in contours]

    # Locate wall contour
    wall_idx = np.argmax(areas)
    wall_hierarchy = hierarchy[wall_idx]
    wall_contour = contours[wall_idx]

    # Locate stray contours: usually patches that arent on the tip
    wall_area = areas[wall_idx]
    stray_area_thresh = wall_area * stray_area_thresh_ratio
    stray_contours = [contours[i] for i in range(len(contours)) if
        areas[i] > stray_area_thresh and  # larger than threshold,
        i != wall_idx and  # not the identified wall,
        hierarchy[i][3] == -1  # not inside another contour
    ]

    # Locate hole contour
    hole_contour = None
    largest_child_area = 0
    child_idx = wall_hierarchy[2]
    while child_idx != -1:
        child_area = areas[child_idx]
        if child_area > largest_child_area:
            largest_child_area = child_area
            hole_contour = contours[child_idx]

        child_idx = hierarchy[child_idx][0]

    # After preprocessing, the resulting hole is always a contour inside a contour.
    # for i in range(len(hierarchy)):
    #     hierarchy_datum = hierarchy[i]
    #     child = hierarchy_datum[2]
    #     parent = hierarchy_datum[3]
    #     # print(hierarchy_datum)
    #     if child == -1 and parent != -1:
    #         hole_contour = contours[i]
    #         break

    return wall_contour, hole_contour, stray_contours,


def get_contour_center(contour):
    M = cv.moments(contour)
    centroid = (
            int(M["m10"] / M["m00"]),
            int(M["m01"] / M["m00"])
        )
    return centroid


def create_circle_contour(radius):
    mat = np.zeros(
        (radius * 2 + 10, radius * 2 + 10),
        dtype=np.uint8
    )
    cv.circle(mat, (radius + 5, radius + 5), radius, 255, thickness=-1)
    contours, _ = cv.findContours(
        mat, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[-2:]

    return contours[0]


def check_circularity(contour):
    area = cv.contourArea(contour)
    # center = get_contour_center(contour)
    radius = (area / np.pi) ** 0.5
    circle_contour = create_circle_contour(int(radius))

    mat = np.zeros(
        (500, 500),
        dtype=np.uint8
    )
    cv.drawContours(mat, [circle_contour, contour], -1, 255, 1)

    return cv.matchShapes(contour, circle_contour, method=1, parameter=0.0)


def get_metrics(
        img_path,
        locate_hole_thresh=80,
        locate_defect_thresh=55,
        tip_area=250,
        image_resize=0.5):
    #
    # ------------------------- READ AND PREPROCESS IMAGE
    #
    img = read_image(img_path, image_resize)
    img = cv.equalizeHist(img)
    _, bin_img = cv.threshold(img, thresh=locate_hole_thresh,
                          maxval=255, type=cv.THRESH_BINARY)

    #
    # ------------------------- LOCATE HOLE
    #
    # Denoise
    cleaned_img = remove_small_contours(bin_img, 200)
    cleaned_img = cv.morphologyEx(
        cleaned_img, cv.MORPH_CLOSE, kernel_gethole_close)
    cleaned_img = cv.morphologyEx(
        cleaned_img, cv.MORPH_OPEN, kernel_gethole_open)

    # Locate center dot
    _, hole_cnt, _ = locate_interest_contours(cleaned_img)
    if hole_cnt is None:
        return None

    hole_pos = get_contour_center(hole_cnt)

    # Crop image
    interest_area_img = crop_image(img, hole_pos, tip_area)
    interest_area_img = apply_mask(interest_area_img, mask)
    interest_area_img = cv.equalizeHist(interest_area_img)

    #
    # ------------------------- MARK IMPERFECTIONS
    #
    blurred_img = blur(interest_area_img, 7)
    _, cropped_bin_img = cv.threshold(blurred_img, thresh=locate_defect_thresh,
                          maxval=255, type=cv.THRESH_BINARY)
    morphed_img = cv.morphologyEx(
        cropped_bin_img, cv.MORPH_CLOSE, kernel_noiseclear_close)
    morphed_img = cv.morphologyEx(
        cropped_bin_img, cv.MORPH_OPEN, kernel_noiseclear_open)

    # Patch stray contours
    wall_cnt, hole_cnt, stray_cnt = locate_interest_contours(morphed_img)
    cv.drawContours(morphed_img, stray_cnt, contourIdx=-1,
                      color=0, thickness=-1)

    # CHECKPOINT: Check inner hole defect
    convex_hole_contour = cv.convexHull(hole_cnt)
    hole_defect_mask = np.zeros_like(morphed_img, dtype=np.uint8)
    cv.drawContours(hole_defect_mask, [convex_hole_contour], contourIdx=-1,
                      color=255, thickness=-1)
    masked_hole_defect_img = np.bitwise_and(hole_defect_mask, morphed_img)
    hole_defect_score = (np.sum(masked_hole_defect_img) / 255)

    # CHECKPOINT: Check hole coencentricity
    hole_circdev_score = check_circularity(hole_cnt) * 1e4

    # CHECKPOINT: Check pipette wall defects
    convex_wall_contour = cv.convexHull(wall_cnt)
    unholed_img = morphed_img.copy()
    cv.drawContours(unholed_img, [hole_cnt], contourIdx=-1,
                      color=255, thickness=-1)

    wall_defect_mask = np.zeros_like(unholed_img, dtype=np.uint8)
    cv.drawContours(wall_defect_mask, [convex_wall_contour], contourIdx=-1,
                      color=255, thickness=-1)

    masked_wall_defect_img = np.bitwise_xor(wall_defect_mask, unholed_img)
    wall_defect_score = (np.sum(masked_wall_defect_img) / 255)

    # CHECKPOINT: Check pipette wall coencentricity
    wall_circdev_score = check_circularity(wall_cnt) * 1e4

    metrics = (
        wall_defect_score,
        wall_circdev_score,
        hole_defect_score,
        hole_circdev_score,)
    masks = (
        wall_cnt,
        hole_cnt,
        masked_hole_defect_img,
        masked_wall_defect_img
        )

    return (
        interest_area_img,
        metrics,
        masks
        )


# Default parameters
tip_area = 250
mask = circle_mask(tip_area)
kernel_gethole_close = kernel(25)
kernel_gethole_open = kernel(15)
kernel_noiseclear_close = circ_kernel(3)
kernel_noiseclear_open = circ_kernel(5)
