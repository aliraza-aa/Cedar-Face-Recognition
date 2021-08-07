import math
from functions import read_encodings
from sklearn import neighbors
import os
import os.path
import pickle
import face_recognition
import cv2
import numpy as np

model_path = 'model'
min_confidence = 0.8
prototxt = "models/deploy.prototxt.txt"
model = "models/res10_300x300_ssd_iter_140000.caffemodel"

def predict(img, face_locations, knn_clf, distance_threshold=0.6):

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(img, known_face_locations=face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), face_locations, are_matches)]

def detect_face_image(net, min_confidence, image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))

    # print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()
    no_detections = 0
    locations = []
    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]
        if confidence > min_confidence:
            no_detections += 1
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            locations.append((startY, endX, endY, startX))
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # cv2.imshow("Output", image)
    return image, no_detections, locations

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

with open(model_path, 'rb') as f:
    knn_clf = pickle.load(f)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    x = cv2.waitKey(1)

    frame, no_detections, locs = detect_face_image(net, min_confidence, frame)
    if no_detections >= 1:
        result = predict(frame, locs, knn_clf)
        for (pred, loc) in result:
            cv2.rectangle(frame, (loc[3], loc[0]), (loc[1], loc[2]), [255, 0, 0], 2)
            cv2.rectangle(frame, (loc[3], loc[2]), (loc[1], loc[2]+22), [255, 0, 0], cv2.FILLED)
            cv2.putText(frame, pred, (loc[3]+10, loc[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [240, 240, 240], 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

cap.release()
cv2.destroyAllWindows()