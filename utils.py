from urllib.request import urlretrieve
import cv2 as cv
import matplotlib.pyplot as plt

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

def display_images(img_list, titles):
    """Function to display multiple images
    Arguments:
    ---
    img_list: list of images
    titles: list of titles
    
    Return:
    ---
    One or more display windows"""
    for img, title in zip(img_list, titles):
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