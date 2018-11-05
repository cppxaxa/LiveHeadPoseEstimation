import cv2
import numpy as np

import imutils
import dlib
from imutils import face_utils

from PoseEstimationLib import *
 
# Read Image
# cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# 3D model points.
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
])

poseProjector = PoseProjector(model_points)

while True:
    im = cv2.imread("headPose.jpg")
    #_, im = cap.read()

    image = imutils.resize(im, width=500)
    im = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    #2D image points. If you change the image, you need to change vector
    image_points = np.array([
                                (359, 391),     # Nose tip
                                (399, 561),     # Chin
                                (337, 297),     # Left eye left corner
                                (513, 301),     # Right eye right corne
                                (345, 465),     # Left Mouth corner
                                (453, 469)      # Right mouth corner
                            ], dtype="double")

    if len(rects) <= 0:
        continue

    rect = rects[0]
    # determine the facial landmarks for the face region, then
    # convert the landmark (x, y)-coordinates to a NumPy array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    clone = image.copy()

    # Nose tip
    (x, y) = shape[30]
    image_points[0] = (x, y)
    
    # Chin
    (x, y) = shape[8]
    image_points[1] = (x, y)
    
    # Left eye corner
    (x, y) = shape[36]
    image_points[2] = (x, y)

    # Right eye corner
    (x, y) = shape[45]
    image_points[3] = (x, y)

    # Left mouth corner
    (x, y) = shape[48]
    image_points[4] = (x, y)

    # Right mouth corner
    (x, y) = shape[54]
    image_points[5] = (x, y)

    nose_end_point2D = poseProjector.ProjectPoints((0.0, 0.0, 500.0), im, image_points)
    
    for p in image_points:
        cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
    
    
    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    
    cv2.line(im, p1, p2, (255,0,0), 2)
    
    # Display image
    cv2.imshow("Output", im)
    if cv2.waitKey() == ord('q'):
        break
    