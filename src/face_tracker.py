import numpy as np
import cv2

HAAR_CASCADE_FILE = "haarcascade_frontalface_default.xml"
OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
        }
STATE_DETECT = 0
STATE_TRACK = 1

class FaceTracker():
    def __init__(self, tracker_impl):
        self.state = STATE_DETECT
        self.tracker_impl = tracker_impl
        self.tracker = OBJECT_TRACKERS[tracker_impl]()
        self.classifier = cv2.CascadeClassifier(HAAR_CASCADE_FILE)

    def process(self, frame_gray):
        # Track face.
        if self.state == STATE_TRACK:
            success, face = self.tracker.update(frame_gray)
            face = map(int, face)
            if not success:
                print("Lost track.")
                # Fall back to detection.
                self.state = STATE_DETECT

        # Detect face.
        if self.state == STATE_DETECT:
            face = self.detect_face(frame_gray)
            if face is not None:
                # Switch to tracking state for next frame.
                self.tracker = OBJECT_TRACKERS[self.tracker_impl]()
                self.tracker.init(frame_gray, face)
                self.state = STATE_TRACK
        return face


    def detect_face(self, frame):
        faces = self.classifier.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))#, flags=cv2v.CV_HAAR_SCALE_IMAGE)
        if len(faces) > 0: return tuple(faces[0])
        else: return None
