# README #

A face interface class implementing different face detection, alignment and recognition algorithms.

### Algorithms Implemented ###

* Face Detector (detector_method='dlib')
* Dlib CNN Face Detector (detector_method='cnn')
* OpenCV Face Detector (detector_method='opencv')
* Mobilenet Face Detector (detector_method='mobilenet')
* Dlib Face Recognition (recognition_method='dlib')
* Dlib Facial Landmarks (predictor_model='small' for 5 face landmarks)

### Requirements ###

* dlib
* opencv
* numpy
* cudnn (for gpu supoort for cnn methods)

### Installation ###
```sh
sudo apt-get install libboost-all-dev libopenblas-dev liblapacke-dev cmake build-essential
sudo apt-get install python-dev python-pip python-setuptools #python-opencv
```
```sh
pip install --user git+<https-url>
```
or
```sh
pip install --user git+ssh://git@bitbucket.org/macherlabs/facelib.git
```
### How to use ###

    import face, cv2
    facedemo = face.Face(detector_method='dlib')
    image_url1 = 'test.png'
    image_url2 = 'test2.png'
    
    imgcv1 = cv2.imread(image_url1)
    imgcv2 = cv2.imread(image_url2)

    if imgcv1 is not None and imgcv2 is not None:
        results = facedemo.compare(imgcv1, imgcv2)
        print results# facelib
 
### Mobilenet usage ###

    faceDetector = Face(detector_method='mobilenet',
             detector_model='mobilenet_300_frozen_trt_inference_graph_face.pb', # Default mobilenet_512_frozen_inference_graph_face.pb
             recognition_method=None ,
             gpu_frac=0.3,
             trt_enable=False,# converts graph to tensorrt optimized graph
             precision='FP16' # Defalut 'FP32', use 'FP16' only if gpu support is available
             )
