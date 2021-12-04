import cv2

face_detector = cv2.CascadeClassifier('F:/projects/ml/ml_projects/face_recognition_opencv/haarcascade_frontalface_default.xml')

model = cv2.face_LBPHFaceRecognizer.create()

model.read('Final_model.xml')

name = {1:'Naveen'}

def facedetector(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = face_detector.detectMultiScale(grey, scaleFactor=1.35, minNeighbors=5)
    roi = grey

    if face == ():
        return None

    for x, y, w, h in face:
        roi = grey[y:y+h, x:x+w]
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
        roi = cv2.resize(roi, (500, 500))

    return roi

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    face = facedetector(frame)

    if face is not None:
        Id, result = model.predict(face)
        if result < 500:
            confidence = int((1 - result / 300) * 100)
        if confidence > 85:
            cv2.putText(frame, str(confidence)+"% matched with "+name[Id], (150, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Face Cropper", frame)
        else:
            cv2.putText(frame, 'Unknown Face', (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Face Cropper", frame)

    else:
        cv2.putText(frame, 'No Face Found', (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Face Cropper", frame)
        pass

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()