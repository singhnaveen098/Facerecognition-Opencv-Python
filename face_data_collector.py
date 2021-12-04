import cv2
from pathlib import Path

face_detector = cv2.CascadeClassifier('F:/projects/ml/ml_projects/face_recognition_opencv/haarcascade_frontalface_default.xml')

def facedetector(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = face_detector.detectMultiScale(grey, scaleFactor=1.35, minNeighbors=5)
    roi = grey

    if face == ():
        return None

    for x, y, w, h in face:
        roi = grey[y:y+h, x:x+w]
        # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
        roi = cv2.resize(roi, (500, 500))

    return roi

ID = input('Enter your ID : ')
directory = Path('F:/projects/ml/ml_projects/face_recognition_opencv/facedata/' + str(ID))
if directory.exists():
    print("Id already exists!!")
    overwrite = input('Do you want to overwrite face data? (y/n) ')
    if(overwrite=='n' or overwrite=='N'):
        exit()

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    face = facedetector(frame)
    if face is not None:
        count += 1
        filename_path = 'F:/projects/ml/ml_projects/face_recognition_opencv/facedata/' + str(ID) + '/user.' + str(count) + '.jpg'
        cv2.imwrite(filename_path, face)
        cv2.imshow('Live', face)
    else:
        pass

    if count >= 100:
        break

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
