import cv2
import numpy as np
import os
import face_recognition


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


parentdir = r"C:\Users\pc\Desktop\cedar-face-recog\TrainData"
trainpath = 'TrainData'
current_person = ''
name_switch = 0
record_switch = 0
image_id = 0
model = 'hog'
message = ''

min_confidence = 0.5
prototxt = "models/deploy.prototxt.txt"
model = "models/res10_300x300_ssd_iter_140000.caffemodel"

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

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

    frame2 = frame.copy()

    if x == ord('1'): # quit
        break
    elif x == ord('2'): # record name
        name_switch = 1
        continue
    elif x == ord('3'): # record data
        name_switch = 0
        path = os.path.join(parentdir, current_person)
        try:
            os.mkdir(path)
            record_switch = 1
        except:
            print(f"Can't create a new directory, person's name: {current_person}")
            message = f"cant create a directory: {current_person}"
    elif x == ord('4'):
        current_person = ''
        record_switch = 0
        name_switch = 0
        image_id = 0
        message = ''

    if name_switch == 1:
        if x != -1:
            current_person += chr(x)
    
    if record_switch == 1:
        # face detection through face_recogntion module
        # locs = face_recognition.face_locations(frame, model=model)
        # for loc in locs:
        #     cv2.rectangle(frame, (loc[3], loc[0]), (loc[1], loc[2]), [255, 0, 0], 2)

        #face detection through opencvdl/caffe model (significantly faster and more accurate)
        frame, no_detections, locs = detect_face_image(net, min_confidence, frame)
        if no_detections == 1:
            cv2.imwrite(f"{trainpath}/{current_person}/{image_id}.png", frame2)
            image_id += 1

    (h, w) = frame.shape[:2]

    cv2.putText(frame, f"Current Person: {current_person}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 0], 2)
    cv2.putText(frame, message, (w-350, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)
    cv2.putText(frame, f"Record Switch: {record_switch}", (20, h-35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 2)
    cv2.putText(frame, f"Name Switch: {name_switch}", (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 2)
    cv2.putText(frame, "Key", (w-200, h-75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)
    cv2.putText(frame, "1: Quit", (w-200, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)
    cv2.putText(frame, "2: New Person", (w-200, h-45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)
    cv2.putText(frame, "3: Start Recording", (w-200, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)
    cv2.putText(frame, "4: Reset", (w-200, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)

    cv2.imshow('frame', frame)

cap.release()
cv2.destroyAllWindows()
