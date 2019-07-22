import dlib
import numpy as np
import cv2
class DlibLandmarkDetector():

    def __init__(self, model_path):
        self.face_landmark_detector = dlib.shape_predictor(model_path)

    def detect(self, image, bb):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dlib_rectangle = dlib.rectangle(int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]) )
        landmarks = self.face_landmark_detector(gray, dlib_rectangle)
        landmarks_np = np.zeros((68,2))
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_np[n,0] = x
            landmarks_np[n,1] = y
        
        return landmarks_np, True