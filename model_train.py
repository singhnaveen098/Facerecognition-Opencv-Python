import cv2
import os
import numpy as np

model = cv2.face_LBPHFaceRecognizer.create()

face = []
Id = []

for path, subdirectory, filenames in os.walk('F:/projects/ml/ml_projects/face_recognition_opencv/facedata'):
    for filename in filenames:
        if filename.startswith('.') or filename == 'Thumbs.db':
            continue
        id = os.path.basename(path)
        img_path = os.path.join(path, filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        face.append(image)
        Id.append(int(id))

model.train(face, np.array(Id))

model.write('Final_model.xml')

print('Model Training Completed!!!!')