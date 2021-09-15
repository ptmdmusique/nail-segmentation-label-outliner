import cv2
import numpy as np

color_map = {
    "hue": {
        "low": 0,
        "high": 255,
        "default": 119,
    },

    "saturation": {
        "low": 0,
        "high": 255,
        "default": 68,
    },

    "value": {
        "low": 0,
        "high": 255,
        "default": 203,
    },
}
# poetry run python color_test.py
DATA_FOLDER = "./analyze/"

img = cv2.imread(DATA_FOLDER + "sample.png")
img = cv2.resize(img, (512, 512), fx=0.5, fy=0.5)
cv2.namedWindow('image')


def callback(key: str, type):
    def setter(value):
        global img
        color_map[key][type] = value

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([color["low"] for color in color_map.values()])
        higher_hsv = np.array([color["high"] for color in color_map.values()])
        mask = cv2.inRange(hsv, lower_hsv, higher_hsv)
        result = cv2.bitwise_and(img, img, mask=mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        contoured = cv2.drawContours(result, contours, -1, (255, 255, 0), 1)
        # print(contours[0].shape)
        # print(np.squeeze(contours[0]))

        # polygon_contours = np.squeeze(contours[0])

        # show thresholded image
        cv2.imshow('image', contoured)
    return setter


# create trackbars for color change
# MIN_VALUE = 0
MAX_VALUE = 255
for key in color_map:
    color = color_map[key]
    default_value = color["default"]
    low_label = key + " low"
    high_label = key + " high"

    cv2.createTrackbar(low_label, 'image', default_value,
                       MAX_VALUE, callback(key, "low"))
    cv2.createTrackbar(high_label, 'image', default_value +
                       1, MAX_VALUE, callback(key, "high"))
cv2.waitKey()
