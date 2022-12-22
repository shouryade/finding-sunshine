import dlib
from glob import glob
import cv2
import numpy as np
import os


face_detector = dlib.get_frontal_face_detector()
face_encoder = dlib.face_recognition_model_v1(
    "models/dlib_face_recognition_resnet_model_v1.dat"
)
shape_predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")


def get_face_rects(image):
    """extract face regions from images"""
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # grayscale the image
    rect = face_detector(g, 1)
    return rect


def get_landmarks(image):
    """return facial landmarks for each detected face region"""
    return [shape_predictor(image, rect) for rect in get_face_rects(image)]


def get_encodings(image):
    """get the encodings of a face using the landmarks provided"""
    return [
        np.array(face_encoder.compute_face_descriptor(image, landmark))
        for landmark in get_landmarks(image)
    ]
