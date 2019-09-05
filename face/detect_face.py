#!/usr/bin/env python

import cv2
import os

WORK_DIR = "/LFS/facelib" #os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = 'models'


def draw_rects(img, faces):
    """
    Draws rectangle around detected faces.
    Arguments:
        img: image in numpy array on which the rectangles are to be drawn
        faces: list of faces in a format given in Face Class
    Returns:
        img: image in numpy array format with drawn rectangles
    """
    for face in faces:
        x1, y1 = face['box']['topleft']['x'], face['box']['topleft']['y']
        x2, y2 = face['box']['bottomright']['x'], face['box']['bottomright']['y']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img


class FaceDetectorOpenCV(object):
    """
    A face detector based on OpenCV Cascade Classifier on Haar features.
    """
    def __init__(self, model_loc=None, min_height_thresh=30, min_width_thresh=30):
        if model_loc is None:
            model_name = 'haarcascade_frontalface_alt2.xml'
            model_loc = os.path.join(WORK_DIR, MODEL_DIR, model_name)
        self.min_h = min_height_thresh
        self.min_w = min_width_thresh
        self.face_cascade = cv2.CascadeClassifier(model_loc)

    def detect_raw(self, imgcv, **kwargs):
        if len(imgcv.shape) > 2:
            imgcv = cv2.cvtColor(imgcv, cv2.COLOR_BGR2GRAY)
        imgcv = cv2.equalizeHist(imgcv)
        return self.face_cascade.detectMultiScale(imgcv, 1.3, minNeighbors=5,
                                                  minSize=(self.min_h, self.min_w))

    def detect(self, imgcv, **kwargs):
        faces = self.detect_raw(imgcv, **kwargs)
        return self._format_result(faces)

    # Format the results
    def _format_result(self, result):
        out_list = []
        for x, y, w, h in result:
            formatted_res = dict()
            formatted_res["class"] = 'face'
            formatted_res["prob"] = 0.99
            formatted_res["box"] = {
                "topleft": {'x': x.item(), 'y': y.item()},
                "bottomright": {'x': (x+w).item(), 'y': (y+h).item()}
                }
            out_list.append(formatted_res)
        return out_list


class FaceDetectorDlib(object):
    """
    A face detector based on dlib HoG features.
    """
    def __init__(self):
        import dlib
        self._detector = dlib.get_frontal_face_detector()

    def detect_raw(self, imgcv, **kwargs):
        upsamples = kwargs.get('upsamples', 1)
        if len(imgcv.shape) > 2:
            imgcv = cv2.cvtColor(imgcv, cv2.COLOR_BGR2GRAY)
        equ = cv2.equalizeHist(imgcv)
        return self._detector(equ, upsamples)

    def detect(self, imgcv, **kwargs):
        faces = self.detect_raw(imgcv, **kwargs)
        return self._format_result(faces)

    # Format the results
    def _format_result(self, result):
        out_list = []
        for res in result:
            formatted_res = dict()
            formatted_res["class"] = 'face'
            formatted_res["prob"] = 0.99
            formatted_res["box"] = {
                "topleft": {'x': res.left(), 'y': res.top()},
                "bottomright": {'x': res.right(), 'y': res.bottom()}
                }
            out_list.append(formatted_res)
        return out_list


class FaceDetectorCNN(object):
    """A face detector based on dlib CNN model."""
    def __init__(self, model_loc=None):
        import dlib
        if not model_loc:
            model_name = 'mmod_human_face_detector.dat'
            model_loc = os.path.join(WORK_DIR, MODEL_DIR, model_name)
        self._detector = dlib.cnn_face_detection_model_v1(model_loc)

    def detect_raw(self, imgcv, **kwargs):
        upsamples = kwargs.get('upsamples', 1)
        return self._detector(imgcv, upsamples)

    def detect(self, imgcv, **kwargs):
        faces = self.detect_raw(imgcv, **kwargs)
        return self._format_result(faces)

    # Format the results
    def _format_result(self, result):
        out_list = []
        for res in result:
            formatted_res = dict()
            formatted_res["class"] = 'face'
            formatted_res["prob"] = res.confidence
            formatted_res["box"] = {
                "topleft": {'x': res.rect.left(), 'y': res.rect.top()},
                "bottomright": {'x': res.rect.right(), 'y': res.rect.bottom()}
                }
            out_list.append(formatted_res)
        return out_list


class FaceDetectorYolo(object):
    def __init__(self, model_name='yolo_tiny.ckpt'):
        from .yolodetect import PersonDetectorYOLOTiny
        model_loc = os.path.join(WORK_DIR, MODEL_DIR, model_name)
        self._detector = PersonDetectorYOLOTiny(model_loc)

    def detect_raw(self, imgcv, **kwargs):
        return self._detector.run(imgcv)

    def detect(self, imgcv, **kwargs):
        faces = self.detect_raw(imgcv, **kwargs)
        return self._format_result(faces)

    # Format the results
    def _format_result(self, result):
        out_list = []
        for (x, y, w, h, p) in result:
            formatted_res = dict()
            formatted_res["class"] = 'face'
            formatted_res["prob"] = p
            formatted_res["box"] = {
                "topleft": {'x': x-w/2, 'y': y-h/2},
                "bottomright": {'x': x+w/2, 'y': y+h/2}
                }
            out_list.append(formatted_res)
        return out_list


if __name__ == '__main__':
    import sys
    import pprint

    detector = FaceDetectorDlib()
    image_url = 'test.png' if len(sys.argv) < 2 else sys.argv[1]
    imgcv = cv2.imread(image_url)
    if imgcv is not None:
        print(imgcv.shape)
        results = detector.detect(imgcv)
        pprint.pprint(results)
        cv2.imshow('Faces', draw_rects(imgcv, results))
        cv2.waitKey(0)
    else:
        print("Could not read image: {}".format(image_url))
