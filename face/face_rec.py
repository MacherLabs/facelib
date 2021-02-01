#!/usr/bin/env python
import os
import cv2
import numpy as np
import re
import tensorflow as tf
WORK_DIR = "/LFS/facelib" #os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = 'models'

class FaceRecFacenet(object):
    def __init__(self, model_loc=None, gpu_fraction=0.8):
        if model_loc is None:
            self.model_name = "20170512-110547.pb"
            self.model_loc = os.path.join(WORK_DIR, MODEL_DIR, self.model_name)

        self.load_model()
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
        self.sess = tf.compat.v1.Session(config=config)
        self.embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
        self.images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
        self.phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
        init = tf.compat.v1.global_variables_initializer()
    
    def prewhiten(self, x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1/std_adj)
        return y

    def load_model(self):
        model_exp = os.path.expanduser(self.model_loc)
        print('Model filename: %s' % model_exp)
        with tf.io.gfile.GFile(model_exp, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

    def get_model_filenames(self):
        files = os.listdir(self.model_loc)
        meta_files = [s for s in files if s.endswith('.meta')]
        if len(meta_files) == 0:
            raise ValueError('No meta file found in the model directory (%s)' % self.model_loc)
        elif len(meta_files) > 1:
            raise ValueError('More than one meta file in the model directory (%s)' % self.model_loc)
        meta_file = meta_files[0]
        meta_files = [s for s in files if '.ckpt' in s]
        max_step = -1
        for f in files:
            step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
            if step_str is not None and len(step_str.groups()) >= 2:
                step = int(step_str.groups()[1])
                if step > max_step:
                    max_step = step
                    ckpt_file = step_str.groups()[0]
        return meta_file, ckpt_file

    def face_encodings(self, face_image, image_size=160):
        img = self.prewhiten(face_image)
        w, h, _ = img.shape
        if (w*h) >= (image_size*image_size):
            resize_img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        else:
            resize_img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        feed_dict = {self.images_placeholder:np.expand_dims(resize_img, axis=0), self.phase_train_placeholder:False}
        return [np.array(self.sess.run(self.embeddings, feed_dict=feed_dict)[0])]

    def face_distance(self,face_encodings, face_to_compare):
        if len(face_encodings) == 0:
            return np.empty((0))
        return np.linalg.norm(face_encodings - face_to_compare, axis=1)

    def face_similarity(self, face_encodings, face_encodings_to_compare, method="distance"):
        if method=="distance":
            if len(face_encodings) == 0:
                return np.empty((0))
            return np.linalg.norm(face_encodings - face_encodings_to_compare, axis=1)
        elif method == "dotproduct":
            ans = np.sum(face_encodings*face_encodings_to_compare, axis=1)
            return ans
        else:
            print("No method specified for comparison!")


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

    def face_similarity(self, face_encodings, face_encodings_to_compare, method="distance"):
        if method=="distance":
            if len(face_encodings) == 0:
                return np.empty((0))
            return np.linalg.norm(face_encodings - face_encodings_to_compare, axis=1)
        elif method == "dotproduct":
            ans = np.sum(face_encodings*face_encodings_to_compare, axis=1)
            return ans
        else:
            print("No method specified for comparison!")
    
