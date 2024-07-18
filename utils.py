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