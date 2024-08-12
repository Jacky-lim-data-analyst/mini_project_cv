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

## Project: Apple detector
### Overview
- **Objective**: Develop and train HOG + Linear SVM pipeline that can detect apples in images.
- **Recommended method**: HOG and Linear SVM
- **Resources**: OpenCV, sklearn, NumPy.

### Dataset
The dataset can be found at [Kaggle](https://www.kaggle.com/datasets/mbkinaci/fruit-images-for-object-detection)

### Deliverable example
<img width="521" alt="image" src="https://github.com/user-attachments/assets/df04361b-bd91-4e74-9392-ee0d7ed4b450">

## Project: Car plate detection
### Overview
- **Objective**: Develop a **robust** car plate detector.
- **Recommended method**: Pretrained YOLO or SSD object detector and dedicated image processing algorithms (edge detection, morphological operations, thresholding, contour detection and etc).
- **Resources**: OpenCV, NumPy.

### Dataset
The dataset can be found at `mini_project_cv/w9/malaysia_car_plate`

### Deliverable example
![res5](https://github.com/user-attachments/assets/6606063a-63e2-4d91-bd07-3492c1c62d2b)

