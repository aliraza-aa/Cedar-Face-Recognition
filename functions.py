import face_recognition, cv2, os, time

attendancelog = []

def encode_images():
    trainpath = "TrainData"
    knownfaces, knownnames = [], []
    for name in os.listdir(trainpath):
        for filename in os.listdir(f"{trainpath}/{name}"):
            img = face_recognition.load_image_file(f"{trainpath}/{name}/{filename}")
            try:
                img = list(face_recognition.face_encodings(img)[0])
                knownfaces.append(img)
                knownnames.append(name)
            except:
                print(name, filename)
    with open("encodings.txt", "w") as f:
        f.write(str((knownfaces, knownnames)))
    print('Encodings Saved')

def read_encodings():
    with open("encodings.txt", "r") as f:
        data = eval(f.read())
    print('Encodings Loaded')
    return data[0], data[1]

def test_against_images(path, model):
    encodings, names = read_encodings()
    for filename in os.listdir(path):
        img = face_recognition.load_image_file(f"{path}/{filename}")
        locs = face_recognition.face_locations(img, model=model)
        encs = face_recognition.face_encodings(img, locs)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for faceenc, faceloc in zip(encs, locs):
            results = list(face_recognition.face_distance(encodings, faceenc))
            match = names[results.index(min(results))]
            print(f"Match found: {match}")
            cv2.rectangle(img, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]), [255, 0, 0], 2)
            cv2.rectangle(img, (faceloc[3], faceloc[2]), (faceloc[1], faceloc[2]+22), [255, 0, 0], cv2.FILLED)
            cv2.putText(img, match, (faceloc[3]+10, faceloc[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [240, 240, 240], 2)
        cv2.imshow(filename, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def logattendance(attendee):
    global attendancelog
    if attendee in attendancelog:
        return
    with open("attendance.txt", 'a') as f:
        now = time.strftime("%m/%d/%Y, %H:%M:%S - ", time.localtime())
        f.write(now + attendee + '\n')
    attendancelog.append(attendee)

def test_against_video(path, model, attendance = False):
    encodings, names = read_encodings()
    video = cv2.VideoCapture(0)
    while True:
        ret, img = video.read()
        if not ret:
            print("can't grab frame")
            break
        locs = face_recognition.face_locations(img, model=model)
        encs = face_recognition.face_encodings(img, locs)
        for faceenc, faceloc in zip(encs, locs):
            results = list(face_recognition.face_distance(encodings, faceenc))
            match = names[results.index(min(results))]
            if attendance: logattendance(match)
            cv2.rectangle(img, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]), [255, 0, 0], 2)
            cv2.rectangle(img, (faceloc[3], faceloc[2]), (faceloc[1], faceloc[2]+22), [255, 0, 0], cv2.FILLED)
            cv2.putText(img, match, (faceloc[3]+10, faceloc[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [240, 240, 240], 2)
        cv2.imshow("Video Feed", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

testpath = "TestData"
model = "hog"

# encode_images()
# test_against_video(2, model, True)