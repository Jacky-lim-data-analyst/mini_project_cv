# Week 8 in-class project

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
Example input image:

![apple image](clock.jpg)

Example of corresponding output segmented image:

![clock jpg_res](https://github.com/user-attachments/assets/09b8f4a4-b553-4369-a4f2-7ff6be9a64d7)
