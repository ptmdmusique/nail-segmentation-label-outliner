import base64
import json
import os
import cv2
import numpy as np

DATA_FOLDER = "./assets/"
INPUT_FOLDER = "./input/"
OUTPUT_FOLDER = "./output/"

label_color_list = [
    {
        "label": "finger_little",
        "color": [64, 128, 0],  # rgb
        # hsv
        "lower": [45, 255, 128],
        "upper": [46, 255, 129],
    },
    {
        "label": "finger_ring",
        "color": [0, 128, 128],  # rgb
        # hsv
        "lower": [90, 255, 128],
        "upper": [92, 255, 130],
    },
    {
        "label": "finger_middle",
        "color": [128, 0, 128],  # rgb
        # hsv
        "lower": [150, 255, 128],
        "upper": [151, 255, 129],
    },
    {
        "label": "finger_index",
        "color": [149, 150, 204],  # rgb
        # hsv
        "lower": [119, 68, 203],
        "upper": [121, 70, 205],
    },
    {
        "label": "finger_thumb",
        "color": [212, 230, 76],  # rgb
        # hsv
        "lower": [34, 171, 228],
        "upper": [35, 172, 231],
    },
]


def create_shape(label, points):
    return {
        "label": label,
        "points": points,
        "group_id": None,
        "shape_type": "polygon",
        "flags": {}
    }


def create_label_file(image_path, shapes):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    name = image_path.split("/")[-1]
    out_object = {
        "version": "4.5.6",
        "flags": {},
        "shapes": shapes,
        "imagePath": "./" + name,
        "imageData": encoded_string
    }

    file_name = os.path.basename(image_path).split(".")[0]
    with open(OUTPUT_FOLDER + file_name + ".json", "w") as out_file:
        json.dump(out_object, out_file)


def create_contour_file(labeled_path):
    image = cv2.imread(labeled_path)
    # image = cv2.resize(image, (512, 512), fx=0.5, fy=0.5)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # cv2.imshow("label", image)

    shapes = []

    for label_color in label_color_list:
        label = label_color["label"]
        lower = np.array(label_color["lower"])
        upper = np.array(label_color["upper"])
        mask = cv2.inRange(hsv, lower, upper)
        # res = cv2.bitwise_and(image, image, mask=mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        # contoured = cv2.drawContours(res, contours, -1, (255, 255, 0), 1)
        # cv2.imshow("label2", mask)
        # cv2.imshow(label, contoured)

        if len(contours) > 0:
            polygon_contours = np.squeeze(contours[0]).tolist()
            if isinstance(polygon_contours[0], list):
                shapes.append(create_shape(label, polygon_contours))

    # if len(shapes) <= 0:
        # shapes.append(create_shape("unknown", [1, 1]))
    if len(shapes) > 0:
        create_label_file(labeled_path, shapes)


# create_contour_file(DATA_FOLDER + "0a3fdc9f65.png")
# cv2.waitKey()
# poetry run python main.py


def analyze_folder():
    for filename in os.listdir(INPUT_FOLDER):
        extension = filename.split(".")[1]
        if extension != "json":
            create_contour_file(INPUT_FOLDER + filename)


analyze_folder()
