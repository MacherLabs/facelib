#!/usr/bin/env python
import os
import numpy as np

WORK_DIR = "/LFS/facelib" #os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = 'models'


class FaceRecDlib(object):
    def __init__(self, model_loc=None):
        import dlib
        if model_loc is None:
            model_name = 'dlib_face_recognition_resnet_model_v1.dat'
            model_loc = os.path.join(WORK_DIR, MODEL_DIR, model_name)
        self.face_encoder = dlib.face_recognition_model_v1(model_loc)

    def face_encodings(self, face_image, raw_landmarks, num_samples=1):
        """
        Given an image, return the 128-dimension face encoding for each face in the image.
        :param face_image: The image that contains one or more faces
        :param known_face_locations: Optional - the bounding boxes of each face if you
            already know them.
        :param num_samples: How many times to re-sample the face when calculating encoding
            Higher is more accurate, but slower (i.e. 100 is 100x slower)
        :return: A list of 128-dimensional face encodings (one for each face in the image)
        """
        return [np.array(self.face_encoder.compute_face_descriptor(
            face_image, raw_landmark_set, num_samples
            )) for raw_landmark_set in raw_landmarks]

    def face_distance(self, face_encodings, face_to_compare):
        """
        Given a list of face encodings, compare them to a known face encoding and get
            a euclidean distance
        for each comparison face. The distance tells you how similar the faces are.
        :param faces: List of face encodings to compare
        :param face_to_compare: A face encoding to compare against
        :return: A numpy ndarray with the distance for each face in the same order as
            the 'faces' array
        """
        if len(face_encodings) == 0:
            return np.empty((0))
        return np.linalg.norm(face_encodings - face_to_compare, axis=1)
