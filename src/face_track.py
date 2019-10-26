import argparse
import time

import numpy as np
import cv2

from face_tracker import FaceTracker

def parse_args():
    parser = argparse.ArgumentParser(description="Detects and tracks a face.")
    parser.add_argument("--input_file", type=str, help="The path of the video file to process.")
    parser.add_argument("--viz", action="store_true", help="Enables visualization of each frame.")
    parser.add_argument("--tracker", type=str, default="kcf", help="Tracking algorithm, e. g. 'kfc' of 'mosse'")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.input_file is not None:
        cap = cv2.VideoCapture(args.input_file)
    else:
        cap = cv2.VideoCapture(0)
    tracker = FaceTracker(args.tracker)
    n_frames = 0
    face = None
    t_start = time.time()
    # Detect face.
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break # No more frames?
        n_frames += 1

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = tracker.process(frame_gray)

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
