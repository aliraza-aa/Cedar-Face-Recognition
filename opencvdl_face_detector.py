# import the necessary packages
import numpy as np
import argparse
import cv2


min_confidence = 0.5
prototxt = "models/deploy.prototxt.txt"
model = "models/res10_300x300_ssd_iter_140000.caffemodel"

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

def detect_face_video(net, min_confidence):

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, image = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
    
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))

        print("[INFO] computing object detections...")
        net.setInput(blob)
        detections = net.forward()
        print(f"No of faces detected: {len(detections)}")

        for i in range(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]
            if confidence > min_confidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
        
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(image, (startX, startY), (endX, endY),
                    (0, 0, 255), 2)
                cv2.putText(image, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        cv2.imshow("Output", image)

        if cv2.waitKey(1) &0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def detect_face_image(net, min_confidence, image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))

    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()
    print(f"No of faces detected: {len(detections)}")
    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]
        if confidence > min_confidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
    
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow("Output", image)

detect_face_video(net, min_confidence)




