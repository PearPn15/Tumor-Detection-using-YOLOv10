from ultralytics import YOLOv10
import cv2
import matplotlib.pyplot as plt
import numpy as np

model_path = "runs\\detect\\train3\\weights\\best.pt"
model = YOLOv10(model_path)

image = cv2.imread('Tr-gl_0475_jpg.rf.ccfac06bc793345d2cc6162e4f3438a5.jpg')
result = model.predict(source='Tr-gl_0475_jpg.rf.ccfac06bc793345d2cc6162e4f3438a5.jpg', conf=0.25, save=True)
for box in result[0].boxes:
    print(box.xyxy,box.cls)
for box in result[0].boxes:
    xmi,ymi,xma,yma=map(int,box.xyxy[0].cpu().np())
    cid=int(box.cls[0].cpu().np())
    cv2.rectangle(image,(xmi,ymi),(xma,yma))
    confidence=box.conf[0].item()
    cv2.putText(image,(xmi,ymi-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,2)
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BAYER_BG2BGR))

    

