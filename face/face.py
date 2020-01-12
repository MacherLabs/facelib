#!/usr/bin/env python

try:
    import dlib
except:
    pass
import os

WORK_DIR = "/LFS/facelib"
MODEL_DIR = 'models'


def bbox_to_dlib_rect(box):
    """
    Convert a box in (left, top, right, bottom) order to a dlib `rect` object
    :param css:  plain tuple representation of the rect in
        (top, right, bottom, left) order
    :return: a dlib `rect` object
    """
    x1, y1 = int(box['topleft']['x']), int(box['topleft']['y'])
    x2, y2 = int(box['bottomright']['x']), int(box['bottomright']['y'])
    return dlib.rectangle(x1, y1, x2, y2)


def size_bbox(box):
    w = box['bottomright']['x'] - box['topleft']['x']
    h = box['bottomright']['y'] - box['topleft']['y']
    return w, h, w*h


class Face(object):
    """
    A face interface class implementing different face detection, alignment and
    recognition algorithms.
    Arguments:
        method: name of the algorithm to identify face (default: opencv)
        model: external model path to use for certain algorithms
            (default: included models)
    Methods:
        detect()
            imgcv = grayscale or colored numpy image
            kwargs = optional argumentts for some algorithms e.g. upsamples=1 for dlib,
            threshold =0.7 for mobilenet
    """
    def __init__(self, detector_method='opencv', detector_model=None,
                 predictor_model='small', recognition_method='dlib',
                 recognition_model=None,trt_enable=False,precision ='FP32',gpu_frac=0.3):
        if detector_method == 'dlib':
            from .detect_face import FaceDetectorDlib
            self._detector = FaceDetectorDlib()
        elif detector_method == 'cnn':
            from .detect_face import FaceDetectorCNN
            self._detector = FaceDetectorCNN(detector_model)
        elif detector_method == 'yolo':
            from .detect_face import FaceDetectorYolo
            self._detector = FaceDetectorYolo()
        elif detector_method == 'mobilenet':
            from .detect_face import FaceDetectorMobilenet
            self._detector = FaceDetectorMobilenet(trt_enable=trt_enable,precision =precision,gpu_frac=gpu_frac)
        else:
            from .detect_face import FaceDetectorOpenCV
            self._detector = FaceDetectorOpenCV(detector_model)

        if predictor_model == 'large':
            pose_predictor = os.path.join(
                WORK_DIR, MODEL_DIR, 'shape_predictor_68_face_landmarks.dat')
        elif predictor_model == 'small':
            pose_predictor = os.path.join(
                WORK_DIR, MODEL_DIR, 'shape_predictor_5_face_landmarks.dat')
        if predictor_model is not None:
            self._predictor = dlib.shape_predictor(pose_predictor)

        if recognition_method == 'dlib':
            from .face_rec import FaceRecDlib
            self._recognizer = FaceRecDlib(recognition_model)

    def detect(self, imgcv, **kwargs):
        return self._detector.detect(imgcv, **kwargs)

    def detect_raw(self, imgcv, **kwargs):
        return self._detector.detect_raw(imgcv, **kwargs)

    def detect_largest(self, imgcv, **kwargs):
        faces = self.detect(imgcv, **kwargs)
        if len(faces) > 0:
            return max(faces, key=lambda rect: size_bbox(rect['box'])[-1])

    def get_landmarks(self, imgcv, face_locations=None):
        if face_locations is None:
            face_locations = self.detect(imgcv)
        face_locations = [bbox_to_dlib_rect(face_location['box'])
                          for face_location in face_locations]
        # print(len(face_locations))
        return [self._predictor(imgcv, face_location) for face_location in face_locations]

    def get_encodings(self, imgcv, landmarks, **kwargs):
        num_samples = kwargs.get('num_samples', 1)
        return self._recognizer.face_encodings(imgcv, landmarks, num_samples)

    def get_distance(self, face_encodings, face_to_compare):
        return self._recognizer.face_distance(face_encodings, face_to_compare)

    def compare(self, face_image1, face_image2, tolerance=0.6):
        face1 = self.detect_largest(face_image1)
        face2 = self.detect_largest(face_image2)
        results = {
                    "face1": face1,
                    "face2": face2,
                    "threshold": tolerance
                    }
        if face1 is not None and face2 is not None:
            encodings = self.get_encodings(
                face_image1, self.get_landmarks(face_image1, [face1]))
            encoding = self.get_encodings(
                face_image2, self.get_landmarks(face_image2, [face2]))
            dist = self.get_distance(encodings, encoding[0])[0]
            results["isMatched"] = bool(dist <= tolerance)
            results["matchingConfidence"] = ((tolerance-dist/2)*100/tolerance if
                                             dist < 2*tolerance else 0.0)
        return results


if __name__ == '__main__':
    import sys
    import cv2
    import pprint

    facedemo = Face(detector_method='dlib')
    if len(sys.argv) < 3:
        print("Give two images as arguments")
    else:
        image_url1 = sys.argv[1]
        image_url2 = sys.argv[2]

    imgcv1 = cv2.imread(image_url1)
    imgcv2 = cv2.imread(image_url2)

    if imgcv1 is not None and imgcv2 is not None:
        results = facedemo.compare(imgcv1, imgcv2)
        pprint.pprint(results)
        cv2.imshow('Face1', imgcv1)
        cv2.imshow('Face2', imgcv2)
        cv2.waitKey(0)
    else:
        print("Could not read image: {}".format(image_url1))
