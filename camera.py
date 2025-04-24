import cv2
import time
import imutils
from imutils.video import VideoStream
from detector import load_face_model, detect
from eval import predict

# load serialized face detection model
faceNet = load_face_model()

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream
    frame = vs.read()
    frame = imutils.resize(frame, width=800)

    # detect faces in the frame
    (locs, faces) = detect(frame, faceNet)

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        preds = predict(faces)
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box
            (startX, startY, endX, endY) = box
            # display the label and bounding box rectangle on the output frame
            label = "Glasses" if pred == 0 else "No Glasses"
            color = (0, 255, 0) if pred == 0 else (0, 0, 255)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()