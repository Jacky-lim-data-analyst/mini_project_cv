# Mini Project (Computer vision)

## Course overview

## Prerequisite
- Basic programming skills (preferably in Python)
- Familiarity with linear algebra and calculus
- Understanding of basic probability and statistics

## Course content
Starting from week 2, the course outline is as follows:
1. Python fundamentals
3. Basic image & video operations
4. Basic concepts of digital image
5. Image processing I
6. Image processing II
7. Edge detection
8. Color-based segmentation
9. Image classification (Traditional ML & DL)
10. Semantic segmentation

## Course materials & References

## Assignment & Projects
- Weekly programming assignments
- Mid-term project: Image processing
- Final project: Developing a complete pipeline / model to tackle CV problems

## Access notebooks in Google colab
- Week 2 notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacky-lim-data-analyst/mini_project_cv/blob/main/w2_tutorial.ipynb)
   - Week 2 Exercise solution: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacky-lim-data-analyst/mini_project_cv/blob/main/w2_tut_ans.ipynb)
- Week 3 notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacky-lim-data-analyst/mini_project_cv/blob/main/w3_tutorial.ipynb)
   - Week 3 Exercise solution: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacky-lim-data-analyst/mini_project_cv/blob/main/w3_tut_ans.ipynb)
- Week 4 notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacky-lim-data-analyst/mini_project_cv/blob/main/w4_tutorial.ipynb)
   - Week 4 Exercise solution: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacky-lim-data-analyst/mini_project_cv/blob/main/w4_tut_ans.ipynb)
- Week 5 notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacky-lim-data-analyst/mini_project_cv/blob/main/w5_tutorial.ipynb)
   - Week 5 Exercise solution: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacky-lim-data-analyst/mini_project_cv/blob/main/w5_tut_ans.ipynb)
- Week 6 notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacky-lim-data-analyst/mini_project_cv/blob/main/w6_tutorial.ipynb)
   - Week 6 Exercise solution: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacky-lim-data-analyst/mini_project_cv/blob/main/w6_tut_ans.ipynb)
- Week 8 notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacky-lim-data-analyst/mini_project_cv/blob/main/w8_demo_project.ipynb)
- Week 9 notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacky-lim-data-analyst/mini_project_cv/blob/main/w9_demo_project.ipynb)
- Week 10 notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacky-lim-data-analyst/mini_project_cv/blob/main/w10_dl_with_keras.ipynb)


<div style="background-color: #f0f8ff; border-left: 6px solid #4682b4; padding: 15px; margin-bottom: 15px;">
  <h3 style="color: #4682b4; margin-top: 0;">📌 Important</h3>
    Starting from week 10, all the notebooks uploaded involves training of deep learning model, which requires GPU computing power. If your device does not have GPU, it is recommended to run the codes in Google Colab.
</div>

---

> **Warning** 🚨
> It should be noted that the execution of `cv.imshow()` in Google Colab will cause exception. The workaround is as follow:

 ```{python}
from google.colab.patches import cv2_imshow
cv2_imshow(image)
```
---
# Week 8 in-class project 

## Project title: Red and green apples detection

### Overview
- **Objective**: segment the red and green apples (objects) in different images.
- **Recommended method**: Color-based approach
- **Resources**: OpenCV

### Dataset
Sample dataset can be found at: `images/w8/apples`

### Deliverable example
Example input image:

![apple image](images/w8/apples/apple1.jpg)

Example of corresponding output segmented image:

![app0_res](https://github.com/user-attachments/assets/20d59b34-6dc4-4b8a-a63c-636b0d7dcfdf)

### Sample solution
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacky-lim-data-analyst/mini_project_cv/blob/main/w8_color_segmentation.ipynb)

## Project title: circles detection

### Overview
- **Objective**: detect circular object(s) in different images.
- **Recommended method**: Shape-based approach
- **Resources**: OpenCV

### Dataset
The dataset can be found at: `images/w8/circles`

### Deliverable example
Example input image:

![apple image](images/w8/circles/clock.jpg)

Example of corresponding output segmented image:

![clock jpg_res](https://github.com/user-attachments/assets/09b8f4a4-b553-4369-a4f2-7ff6be9a64d7)

### Sample solution
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacky-lim-data-analyst/mini_project_cv/blob/main/w8_circle_rect.ipynb)

## Project title: rectangles detection

### Overview
- **Objective**: detect rectangular object(s) in different images
- **Recommended method**: Shape-based approach
- **Resources**: OpenCV

### Dataset
The dataset can be found at: `images/w8/rectangles`

### Deliverable example
Example input image:

![ipad image](images/w8/rectangles/ipad.jpg)

Example of corresponding output segmented image:

![ipad_res](https://github.com/user-attachments/assets/45cb45f7-db09-4230-9940-6f8ace843e86)

### Sample solution
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacky-lim-data-analyst/mini_project_cv/blob/main/w8_circle_rect.ipynb)

## Project title: Document scanner and OCR

### Overview
- **Objective**: perform optical character recognition (OCR) on detected document in different images.
- **Recommended method**: Low-level image processing algorithms: edge detection, perspective transform and OCR engine like Tesseract.
- **Resources**: OpenCV and tesseract. To install tesseract, please refer to this [OverStack forum post](https://stackoverflow.com/questions/46140485/tesseract-installation-in-windows)

#### Installation of Pytesseract on Windows:
To accomplish OCR with Python on Windows, you will need Python and OpenCV which you already have, as well as Tesseract and the Pytesseract Python package.

**To install Tesseract OCR for Windows**:

1. Download the [installer](https://github.com/UB-Mannheim/tesseract/wiki).
2. Run the installer.

**To install and use Pytesseract on Windows**:
1. Simply run `pip install pytesseract`
2. You will need to add the following line in your code in order to be able to call pytesseract on your machine: `pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'` depending on path where you installed your Tesseract OCR.

### Dataset
The dataset can be found at: `images/w8/docs`

### Deliverable example
Be able to print out the text present in an image on console.

### Sample solution
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacky-lim-data-analyst/mini_project_cv/blob/main/w8_document_ocr.ipynb)

---
# Week 9 in-class projects

## Project: Image classification: Santa or not
### Overview
- **Objective**: Train and evaluate ML models to categorize whether an image contains Santa or not.
- **Recommended method**: Traditional machine learning algorithm, which consists of feature extraction and machine learning classifier.
- **Resources**: OpenCV, sklearn, NumPy.

### Dataset
The dataset can be found at [Kaggle](https://www.kaggle.com/datasets/deepcontractor/is-that-santa-image-classification)

### Deliverable example
Given an input image, the model will provide a dichotomous variable: ("not-a-santa", "santa").

### Sample solution
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacky-lim-data-analyst/mini_project_cv/blob/main/w9_ml_classification.ipynb)

## Project: Apple detector
### Overview
- **Objective**: Develop and train HOG + Linear SVM pipeline that can detect apples in images.
- **Recommended method**: HOG and Linear SVM
- **Resources**: OpenCV, sklearn, NumPy.

### Dataset
The dataset can be found at [Kaggle](https://www.kaggle.com/datasets/mbkinaci/fruit-images-for-object-detection)

### Deliverable example
<img width="521" alt="image" src="https://github.com/user-attachments/assets/df04361b-bd91-4e74-9392-ee0d7ed4b450">

### Sample solution
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacky-lim-data-analyst/mini_project_cv/blob/main/w9_obj_det.ipynb)

> You can find the pipeline outputs in `mini_project_cv/images/w9/malaysia_car_plate/results`.

## Project: Car plate detection
### Overview
- **Objective**: Develop a **robust** car plate detector.
- **Recommended method**: Pretrained YOLO or SSD object detector and dedicated image processing algorithms (edge detection, morphological operations, thresholding, contour detection and etc).
- **Resources**: OpenCV, NumPy.

### Dataset
The dataset can be found at `mini_project_cv/w9/malaysia_car_plate`

### Deliverable example
![res5](https://github.com/user-attachments/assets/6606063a-63e2-4d91-bd07-3492c1c62d2b)

### Sample solution
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacky-lim-data-analyst/mini_project_cv/blob/main/w9_obj_det.ipynb)

> You can find the pipeline outputs in `mini_project_cv/obj_det/hog_svm`. To detect car in an image, we can leverage pretrained MobileNet + SSD model with OpenCV `dnn` module. The MobileNet + SSD model is first trained on COCO dataset and then fine-tuned on Pascal VOC dataset. The network architecture (.prototxt) and model weight (.caffemodel) files can be found on this [GitHub repo](https://github.com/djmv/MobilNet_SSD_opencv.git)

---

# Week 10 in-class project 

## Project title: Medical Mnist image classification

### Overview
- **Objective**: train a custom deep learning model to categorize the medical MNIST dataset.
- **Recommended method**: custom CNN classifier
- **Resources**: Tensorflow, Keras, PyTorch and OpenCV. Use [Google Colab](https://colab.google/) or [Kaggle notebook](https://www.kaggle.com/code) for free access of GPU for model training.

### Dataset
Sample dataset can be found on [Kaggle](https://www.kaggle.com/datasets/andrewmvd/medical-mnist)

### Deliverable example
![image](https://github.com/user-attachments/assets/b1bba2ad-b97b-42ba-b792-221374cc373a)

### Sample solution
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacky-lim-data-analyst/mini_project_cv/blob/main/w10_grayscale_img_class_dl.ipynb)

## Project title: Sports ball recognition

### Overview
- **Objective**: Leverage transfer learning (pretrained model fine-tunung) to categorize the types of sports ball.
- **Recommended method**: Pretrained model like ResNet, VGG16, VGG19, MobileNet and etc.
- **Resources**: Tensorflow, Keras, PyTorch and OpenCV. Use [Google Colab](https://colab.google/) or [Kaggle notebook](https://www.kaggle.com/code) for free access of GPU for model training.

### Dataset
Sample dataset can be found on [Kaggle](https://www.kaggle.com/datasets/samuelcortinhas/sports-balls-multiclass-image-classification)

### Deliverable example
![image](https://github.com/user-attachments/assets/caea0106-a77b-4741-8685-47c09bc995bb)

### Sample solution
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacky-lim-data-analyst/mini_project_cv/blob/main/w10_cnn_multiclass_image.ipynb)

## Project title: Denoising "dirty" documents

### Overview
- **Objective**: Train a CNN based deep learning model to clean up the "noisy" scanned documents.
- **Recommended method**: Autoencoder architecture.
- **Resources**: Tensorflow, Keras, PyTorch and OpenCV. Use [Google Colab](https://colab.google/) or [Kaggle notebook](https://www.kaggle.com/code) for free access of GPU for model training.

### Dataset
Sample dataset can be found on [Kaggle](https://www.kaggle.com/competitions/denoising-dirty-documents/data)

### Deliverable example
![image](https://github.com/user-attachments/assets/ae63c7e8-e032-4475-a2d8-f9c8b5da55ae)

### Sample solution
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacky-lim-data-analyst/mini_project_cv/blob/main/w10_autoencoder_denoise.ipynb)

---

# Week 11 in-class project 

## Project title: Segmentation of plantations images

### Overview
- **Objective**: Able to identify and segregate the vegetation regions by applying image segmentation methods.
- **Recommended method**: K-means clustering and watershed segmentation
- **Resources**: OpenCV, scikit-learn, matplotlib

### Dataset
Sample dataset can be found on [Kaggle](https://www.kaggle.com/datasets/trainingdatapro/plantations-segmentation)

### Deliverable example
![image](https://github.com/user-attachments/assets/e5e0994f-513f-4e89-ace8-9913cebadc57)

### Sample solution
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacky-lim-data-analyst/mini_project_cv/blob/main/w11_imageSeg.ipynb)

## Project title: Flood area segmentation with drone images

### Overview
- **Objective**: Able to identify and segregate the flood area segmentation (pixel-level classification) by applying semantic segmentation methods.
- **Recommended method**: Deep learning semantic segmentation techniques like U-Net, SegNet, DeepLabV3+ and etc.
- **Resources**: Tensorflow, Keras, PyTorch and OpenCV. Use [Google Colab](https://colab.google/) or [Kaggle notebook](https://www.kaggle.com/code) for free access of GPU for model training.

### Dataset
Sample dataset can be found on [Kaggle](https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation)

### Deliverable example
![image](https://github.com/user-attachments/assets/58b95877-6bf4-4311-8dc0-b42bdde0cd73)

### Sample solution
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacky-lim-data-analyst/mini_project_cv/blob/main/segment_flood_area.ipynb)

## Project title: Human segmentation

### Overview
- **Objective**: Predict human segmentation mask.
- **Recommended method**: Deep learning semantic segmentation techniques like U-Net, SegNet, DeepLabV3+ and etc.
- **Resources**: Tensorflow, Keras, PyTorch and OpenCV. Use [Google Colab](https://colab.google/) or [Kaggle notebook](https://www.kaggle.com/code) for free access of GPU for model training.

### Dataset
Sample dataset can be found on [Kaggle]([https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation](https://www.kaggle.com/datasets/tapakah68/segmentation-full-body-mads-dataset))

### Deliverable example
![image](https://github.com/user-attachments/assets/fe8b394a-f359-4140-b8ca-d078e49f2de2)

### Sample solution
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacky-lim-data-analyst/mini_project_cv/blob/main/segmentation_human.ipynb)

