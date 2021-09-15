import base64
import json
import os
import cv2
import numpy as np

DATA_FOLDER = "./assets/"

label_color_list = [
    {
        "label": "finger_little",
        "color": [64, 128, 0],  # rgb
        # hsv
        "lower": [45, 255, 128],
        "upper": [46, 256, 129],
    },
    {
        "label": "finger_ring",
        "color": [0, 128, 128],  # rgb
        # hsv
        "lower": [90, 255, 128],
        "upper": [91, 256, 129],
    },
    {
        "label": "finger_middle",
        "color": [128, 0, 128],  # rgb
        # hsv
        "lower": [150, 255, 128],
        "upper": [151, 256, 129],
    },
    {
        "label": "finger_index",
        "color": [149, 150, 204],  # rgb
        # hsv
        "lower": [119, 68, 203],
        "upper": [120, 69, 204],
    },
    {
        "label": "finger_thumb",
        "color": [212, 230, 76],  # rgb
        # hsv
        "lower": [34, 171, 230],
        "upper": [35, 172, 231],
    },
]


# poetry run python main.py


def create_shape(label, points):
    return {"label": label, "points": points}


def create_label_file(image_path, shapes):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    out_object = {
        "version": "4.5.6",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path,
        "imageData": encoded_string
    }

    file_name = os.path.basename(image_path).split(".")[0]
    with open(DATA_FOLDER + file_name + ".json", "w") as out_file:
        json.dump(out_object, out_file)


def get_contour(labeled_path):
    image = cv2.imread(labeled_path)
    image = cv2.resize(image, (512, 512), fx=0.5, fy=0.5)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    shapes = []

    for label_color in label_color_list:
        label = label_color["label"]
        lower = np.array(label_color["lower"])
        upper = np.array(label_color["upper"])
        mask = cv2.inRange(hsv, lower, upper)
        res = cv2.bitwise_and(image, image, mask=mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        contoured = cv2.drawContours(res, contours, -1, (255, 255, 0), 1)
        cv2.imshow("label", hsv)
        cv2.imshow("label2", mask)
        cv2.imshow(label, contoured)

        if len(contours) > 0:
            polygon_contours = np.squeeze(contours[0]).tolist()
            shapes.append(create_shape(label, polygon_contours))

    create_label_file(labeled_path, shapes)


get_contour(DATA_FOLDER + "sample.png")
cv2.waitKey()
