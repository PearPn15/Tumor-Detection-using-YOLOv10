# Tumor-Detection-using-YOLOv10
## Introduction
- Built an AI model using YOLOv10 to detect and classify tumor regions in medical images.
- Customized the Ultralytics YOLO library to adjust input and output pipelines for tumor detection
- Preprocessed and annotated a dataset of CT/MRI scans.
- Fine-tuned YOLOv10 with transfer learning for classification.
- Achieved 90% mAP50 on validation set.
- Deployed a demo web interface for doctors to upload images and receive detection results.
- 🏷️ Report Included
- 📸 Total Images: 3,906 MRI images annotated with bounding boxes in YOLO format.
🏷️ Number of Classes (4 classes):
- Class 0: Glioma
- Class 1: Meningioma
- Class 2: No Tumo
- Class 3: Pituitary Tumor
### Data Split
#### Training Set
- Glioma: 1,153 images
- Meningioma: 1,449 images
- No Tumor: 711 images
- Pituitary Tumor: 1,424 images
#### Validation Set
- Glioma: 136 images
- Meningioma: 140 images
- No Tumor: 100 images
- Pituitary Tumor: 136 images
<img width="1874" height="699" alt="image" src="https://github.com/user-attachments/assets/cb945134-f2a1-411e-b485-4e3c1cc20b3c" />
🔗 YouTube Link: [Project Demo](https://www.youtube.com/watch?v=iM6E3sPSoaQ&t=1s)

## How to use my code
- Dataset : https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes
- Uploaded the annotated MRI dataset (3,906 images, YOLO format) to Roboflow for easier dataset management and version control.
  <img width="1284" height="145" alt="image" src="https://github.com/user-attachments/assets/236739bc-bb2d-480a-bcbc-74af38197099" />
- Replace the path in the file `Classification_Pearpn.ipynb`.
## Requirements
- `python == 3.10.7`
- `opencv-python == 4.8.0.7`
- `torch==2.7.1`
- `pytorch==2.7.1`
## Installation
`git clone https://github.com/PearPn15/Tumor-Detection-using-YOLOv10.git`
