import os
import numpy as np
import cv2 as cv
import json
import argparse
from hole_detector import get_metrics


def convert_to_bgra_color(img, color_channel, alpha=125):
    img_color = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    img_color[:, :, color_channel] = img  # color
    # img_color[:, :, 3] = img  # alpha

    return img_color


def write_on_corner(img, text):
    origin = (5, img.shape[0] - 5)
    return cv.putText(
        img,
        text,
        origin,
        cv.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255, 255),
        1,
        cv.LINE_AA
    )


def get_verdict(metrics, metric_thresholds):
    total_score = 0
    wall_defect_score, wall_circdev_score, hole_defect_score, hole_circdev_score = metrics
    metrics_dict = {
        "hole_defect": hole_defect_score,
        "wall_defect": wall_defect_score,
        "hole_circular_deviation": hole_circdev_score,
        "wall_circular_deviation": wall_circdev_score,
    }

    for key in metrics_dict:
        score = 0
        tested_score = metrics_dict[key]
        weight = metric_thresholds[key]["weight"]
        threshs = metric_thresholds[key]["threshs"]

        for thresh in threshs:
            if tested_score < thresh:
                break

            score += 1

        total_score += score * weight

    verdict = total_score <= metric_thresholds["max_pass_thresh"]
    return (verdict, total_score)


def evaluate(img_path):
    result = get_metrics(
        img_path,
        locate_hole_thresh=80,
        locate_defect_thresh=57,
        tip_area=250,
        image_resize=0.5
        )

    if result == None:
        return False

    img, metrics, masks = result
    wall_defect_score, wall_circdev_score, hole_defect_score, hole_circdev_score = metrics
    wall_cnt, hole_cnt, hole_defect_mask, wall_defect_mask = masks
    verdict, score = get_verdict(metrics, metric_thresholds)

    rgb_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    captioned_rgb_img = rgb_img.copy()
    captioned_rgb_img = write_on_corner(
        captioned_rgb_img,
        "Verdict: {} | {}/{}".format(
            "GO" if verdict else "NG",
            score,
            metric_thresholds["max_pass_thresh"])
    )

    color_hole_defect_mask = convert_to_bgra_color(hole_defect_mask, 2)
    color_wall_defect_mask = convert_to_bgra_color(wall_defect_mask, 1)

    defect_vis_img = rgb_img.copy()
    defect_vis_img = cv.addWeighted(
        defect_vis_img, 1, color_hole_defect_mask, 0.7, -1)
    defect_vis_img = cv.addWeighted(
        defect_vis_img, 1, color_wall_defect_mask, 0.7, -1)
    caption_defect = "Defc: Wall:{:.1f} | Hole:{:.1f}".format(
        wall_defect_score, hole_defect_score
    )
    captioned_defect_img = write_on_corner(
        defect_vis_img,
        caption_defect,
    )

    roundness_vis_img = rgb_img.copy()
    cv.drawContours(roundness_vis_img, [wall_cnt], contourIdx=-1,
                      color=[0, 255, 0, 125], thickness=2)
    cv.drawContours(roundness_vis_img, [hole_cnt], contourIdx=-1,
                      color=[0, 0, 255, 125], thickness=2)
    caption_circularity = "Circ: Wall:{:.1f} | Hole:{:.1f}".format(
        wall_circdev_score, hole_circdev_score
    )
    captioned_roundness_img = write_on_corner(
        roundness_vis_img,
        caption_circularity,
    )

    output_img = np.concatenate((
        captioned_rgb_img,
        captioned_defect_img,
        captioned_roundness_img,
    ), axis=1)

    output_path = img_path.replace(dataset_dir, output_dir)
    # cv.imwrite(output_path, np.bitwise_or(
    #     hole_defect_mask, wall_defect_mask))
    cv.imwrite(output_path, output_img)
    return verdict


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",
                    dest="input",
                    help="Directory to input files. Default: ./dataset",
                    default="dataset")
parser.add_argument("-o", "--output",
                    dest="output",
                    help="Directory to input files. Default: ./output",
                    default="output")
parser.add_argument("-c", "--config",
                    dest="config",
                    help="Configuration file for scoring. Default: ./config.json",
                    default="config.json")
args = parser.parse_args()
dataset_dir = os.path.join(os.getcwd(), args.input)
output_dir = os.path.join(os.getcwd(), args.output)
config_file = os.path.join(os.getcwd(), args.config)

metric_thresholds = json.load(open(config_file, "r"))
img_paths = [os.path.join(dataset_dir, img) for img in os.listdir("dataset")]
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for img_path in img_paths:
    verdict = evaluate(img_path)
    print(img_path, "\t", "GO" if verdict else "NG")
