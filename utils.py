from urllib.request import urlretrieve
import cv2 as cv

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