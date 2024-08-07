import cv2 as cv
import argparse
import numpy as np
from utils import display_images

ap = argparse.ArgumentParser()
# ap.add_argument("-img", "--imgPath", type=str, help="path of source image")
ap.add_argument("-low", "--blueLow", nargs="+", type=int)
ap.add_argument("-high", "--blueHigh", nargs="+", type=int)
args = vars(ap.parse_args())

def color_segment(img: np.ndarray, low_col: tuple, high_col: tuple, smooth: bool = True, 
                  morphology: bool = True) -> np.ndarray:
    img_copy = img.copy()

    if smooth:
        img = cv.GaussianBlur(img, (5, 5), 0)

    img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    mask = cv.inRange(img_hsv, low_col, high_col)
    # morphological operation
    if morphology:
        opened = cv.morphologyEx(mask, cv.MORPH_OPEN, None)
    return cv.bitwise_and(img_copy, img_copy, mask=opened)

# imgs = []
blue_low = tuple(args["blueLow"])
blue_high = tuple(args["blueHigh"])
for i in range(1, 6):
    img_pant = cv.imread(cv.samples.findFile(f"images/color_spaces/pant{i}.jfif"))
    img_pant_rgb = cv.cvtColor(img_pant, cv.COLOR_BGR2RGB)
    res = color_segment(img_pant_rgb, blue_low, blue_high)
    res = cv.cvtColor(res, cv.COLOR_RGB2BGR)
    display_images([img_pant, res], ("original", "segment"))
    cv.imwrite(f"tmp/res{i}.jpg", res)

    