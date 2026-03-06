import cv2
import numpy as np

def preprocess_frame(frame, size=64):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (size, size))
    frame = frame.astype(np.float32) / 255.0
    return frame