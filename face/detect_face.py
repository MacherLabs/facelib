#!/usr/bin/env python

import cv2
import os
import numpy as np

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
        dlib.cuda.set_device(0)
        if not model_loc:
            model_name = 'mmod_human_face_detector.dat'
            model_loc = os.path.join(WORK_DIR, MODEL_DIR, model_name)
        self._detector = dlib.cnn_face_detection_model_v1(model_loc)

    def detect_raw(self, imgcv, **kwargs):
        upsamples = kwargs.get('upsamples', 0)
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

class FaceDetectorMobilenet(object):
    def __init__(self, model_name='mobilenet_512_frozen_inference_graph_face.pb',trt_enable=False,precision ='FP32',gpu_frac=0.3):
        import tensorflow as tf
        import tensorflow.contrib.tensorrt as trt
        model_loc = os.path.join(WORK_DIR, MODEL_DIR, model_name)
        self.detection_graph = tf.Graph()

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_loc, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                # Convert to tensorrt graph if asked
                if trt_enable == True:
                    trt_graph_def=trt.create_inference_graph(input_graph_def= od_graph_def,
                                                max_batch_size=1,
                                                max_workspace_size_bytes=1<<25,
                                                precision_mode=precision,
                                                minimum_segment_size=5,
                                                maximum_cached_engines=5,
                                                outputs=['detection_boxes','detection_scores','detection_classes','num_detections'])

                    tf.import_graph_def(trt_graph_def, name='')
                else:
                    tf.import_graph_def(od_graph_def, name='')

        with self.detection_graph.as_default():
            config = tf.ConfigProto()
            #config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = gpu_frac
            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.windowNotSet = True


    def run(self, image):
        """image: bgr image
        return (boxes, scores, classes, num_detections)
        """
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        #start_time = time.time()
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        #elapsed_time = time.time() - start_time
        #print('inference time cost: {}'.format(elapsed_time))

        return (boxes, scores, classes, num_detections)

    def detect_raw(self, imgcv, **kwargs):
        return self.run(imgcv)

    def detect(self, imgcv, **kwargs):
        threshold = kwargs.get('threshold', 0.7)
        faces = self.detect_raw(imgcv, **kwargs)
        im_height,im_width,_ = imgcv.shape
        return self._format_result(faces,threshold,im_width,im_height)

    # Format the results
    def _format_result(self, result,threshold,im_width,im_height):
        out_list = []
        boxes,scores,classes,num_detections = result
        indexes = np.squeeze(np.argwhere(scores[0]>threshold),axis=1)
        #print("indexes",indexes)
        for index_face in indexes:
            box = boxes[0][index_face]
            ymin, xmin, ymax, xmax = box[0],box[1],box[2],box[3]
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
            prob = scores[0][index_face]
            if prob> threshold:
                formatted_res = dict()
                formatted_res["class"] = 'face'
                formatted_res["prob"] = prob
                formatted_res["box"] = {
                    "topleft": {'x': int(left), 'y': int(top)},
                    "bottomright": {'x': int(right), 'y': int(bottom)}
                    }
                out_list.append(formatted_res)
        #Return the result
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
