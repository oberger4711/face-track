import argparse
import time

import numpy as np
import cv2

HAAR_CASCADE_FILE = "haarcascade_frontalface_default.xml"
OPENCV_OBJECT_TRACKERS = {
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

def parse_args():
    parser = argparse.ArgumentParser(description="Detects and tracks a face.")
    parser.add_argument("input_file", type=str, help="The path of the video file to process.")
    parser.add_argument("--viz", action="store_true", help="Enables visualization of each frame.")
    parser.add_argument("--tracker", type=str, default="kcf", help="Tracking algorithm, e. g. 'kfc' of 'mosse'")
    return parser.parse_args()

def detect_face(frame):
    classifier = cv2.CascadeClassifier(HAAR_CASCADE_FILE)
    faces = classifier.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))#, flags=cv2v.CV_HAAR_SCALE_IMAGE)
    if len(faces) > 0: return tuple(faces[0])
    else: return None

def main():
    args = parse_args()
    tracker = OPENCV_OBJECT_TRACKERS[args.tracker]()
    cap = cv2.VideoCapture(args.input_file)
    n_frames = 0
    state = STATE_DETECT
    face = None
    t_start = time.time()
    # Detect face.
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break # No more frames?
        n_frames += 1
        # Preprocess frame.
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Track face.
        if state == STATE_TRACK:
            success, face = tracker.update(frame_gray)
            face = map(int, face)
            if not success:
                print("Tracking failed.")
                # Fall back to detection.
                state = STATE_DETECT

        # Detect face.
        if state == STATE_DETECT:
            face = detect_face(frame_gray)
            if face is not None:
                # Switch to tracking state for next frame.
                tracker.init(frame_gray, face)
                state = STATE_TRACK

        # Viz.
        if args.viz:
            if face is not None:
                (x, y, w, h) = face
                cv2.rectangle(frame_gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow('Frame', frame_gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    # Track face.
    t_end = time.time()
    fps = n_frames / (t_end - t_start)
    print("FPS: {}".format(fps))
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
