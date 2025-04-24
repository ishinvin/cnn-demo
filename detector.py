import cv2
import numpy as np
from PIL import Image

def load_face_model():
    print('[INFO] loading face detector model...')
    model_path = './pretrained/face_model/res10_300x300_ssd_iter_140000.caffemodel'
    config_path = './pretrained/face_model/deploy.prototxt'
    return cv2.dnn.readNet(model_path, config_path)

def detect(frame, face_net, min_confidence=0.5):
    # grab the dimensions of the frame and then construct a blob from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    face_net.setInput(blob)
    detections = face_net.forward()

    # initialize our list of faces, their corresponding locations,
    faces, locs = [], []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > min_confidence:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI
            face = frame[startY:endY, startX:endX]
            face = Image.fromarray(face).convert("RGB")
            
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    return locs, faces