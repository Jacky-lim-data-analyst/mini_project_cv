from urllib.request import urlretrieve
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def download_save_img(url, filename):
    """Function to download image and save the image as filename
    Arguments:
    ---
    url: URL link
    filename: Image file name
    
    Return:
    ---
    Image file saved on local disk"""
    urlretrieve(url, filename)

def display_image(window_name, image, adjust=False):
    """ Display one image
    Arguments:
    ---
    window_name: str
    image: NumPy array
    adjust: fit the image to monitor size (boolean)
    
    Return:
    ---
    A window showing an image"""
    if adjust:
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    else:
        cv.namedWindow(window_name)
    cv.imshow(window_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def display_images(img_list, titles, adjust=False):
    """Function to display multiple images
    Arguments:
    ---
    img_list: list of images
    titles: list of titles
    adjust: Boolean that triggers cv.WINDOW_NORMAL
    
    Return:
    ---
    One or more display windows"""
    for img, title in zip(img_list, titles):
        if adjust:
            cv.namedWindow(title, cv.WINDOW_NORMAL)
            cv.imshow(title, img)
        else:
            cv.imshow(title, img)

    cv.waitKey(0)
    cv.destroyAllWindows()

def matplotlib_show_images(arr_list, nrow, ncol, titles=[], figsize=(10, 10), axes=False):
    """Function to display multiple images.
    Arguments:
    ---
    arr_list: list of NumPy array images
    nrow, ncol: number of rows and columns (integer)
    titles: list (default: empty)
    figsize: tuple
    axes: boolean
    
    Return:
    ---
    plot"""
    plt.figure(figsize=figsize)
    total_plot = nrow * ncol
    for i in range(total_plot):
        plt.subplot(nrow, ncol, i+1)
        if len(arr_list[i].shape) == 3:
            plt.imshow(arr_list[i])
        elif len(arr_list[i].shape) == 2:
            plt.imshow(arr_list[i], cmap="gray", vmin=0, vmax=255)
        
        if titles:
            plt.title(titles[i])
        
        if not axes:
            plt.axis("off")

    plt.show()

def point_op(img, alpha, beta):
    """Pixel transform function
    Arguments:
    ---
    img: source image (uint8)
    alpha: coefficient
    beta: bias
    
    Returns:
    ---
    image (uint8)"""
    img = img.astype("float32")
    res = alpha * img + beta
    res = np.clip(res, 0, 255)
    return np.uint8(res)

# gamma correction
def gamma_correction(img, gamma=1):
    """Gamma correction function
    Arguments:
    ---
    img: source image (uint8)
    gamma: 1 (default)
    
    Returns:
    ---
    image (uint8)"""
    lookUpTable = np.empty((1, 256), dtype=np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255, gamma) * 255.0, 0, 255)
    return cv.LUT(img, lookUpTable)

# automatic Canny edge detector
def auto_canny(img, method=None, sigma=0.33):
    """Automatic Canny edge detection
    img: grayscale image (uint8)
    methods: otsu, triangle and median (string)
    sigma: default: 0.33
    
    Returns:
    ---
    8-bit single channel image"""
    if method == "median":
        retVal = np.median(img)
    
    elif method == "triangle":
        retVal = cv.threshold(img, 0, 255, cv.THRESH_TRIANGLE)[0]
    
    elif method == "otsu":
        retVal = cv.threshold(img, 0, 255, cv.THRESH_OTSU)[0]
    
    else:
        raise Exception("method specified not available")
    
    lowTh = (1 - sigma) * retVal
    highTh = (1 + sigma) * retVal
    return cv.Canny(img, lowTh, highTh)

def resize_aspect_ratio(img, width=500, interpolation=cv.INTER_LINEAR):
    """Resize image with consistent aspect ratio
    Arguments:
    ---
    img: source image (uint8)
    width: user defined width (int)
    interpolation: `interpolation` argument of cv.resize()
    
    Returns:
    ---
    destination image (uint8)"""
    f = width / img.shape[1]
    return cv.resize(img, None, fx=f, fy=f, interpolation=interpolation)

def nms(bounding_boxes, confidence_score, threshold=0.5):
    """Perform non-maximal suppression:
    Arguments:
    ---
    bounding_boxes: list of bounding boxes in (x1, y1, x2, y2)
    confidence_score: list of scores associated with each bounding box
    threshold: IOU threshold (default=0.5)
    Returns:
    ---
    list of remaining bounding boxes with their confidence scores"""
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = start_x + boxes[:, 2]
    end_y = start_y + boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score

def computeIOUandDice(boxA, boxB):
    """IOU computation
    Arguments:
    ---
    boxA: bounding box in (x1, y1, x2, y2)
    boxB: same format as boxA
    
    Returns:
    ---
    Scalar IOU value and Dice coefficient"""
    x_start = max(boxA[0], boxB[0])
    y_start = max(boxA[1], boxB[1])
    x_end = min(boxA[2], boxB[2])
    y_end = min(boxA[3], boxB[3])

    intersection_area = max(0, (x_end-x_start+1)) * max(0, (y_end-y_start+1))

    # area of bounding boxes
    areaA = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    areaB = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    IOU = intersection_area / (areaA + areaB - intersection_area)
    Dice = 2 * (intersection_area) / (areaA + areaB)
    return IOU, Dice