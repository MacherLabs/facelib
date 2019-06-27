from face import Face
import cv2
import common

url = 0
DETECTOR = 'cnn'
facedemo = Face(detector_method=DETECTOR)


def draw_faces(img, faces):
    """ Draws bounding boxes of objects detected on given image """
    h, w = img.shape[:2]
    for face in faces:
        # draw rectangle
        x1, y1 = face['box']['topleft']['x'], face['box']['topleft']['y'],
        x2, y2 = face['box']['bottomright']['x'], face['box']['bottomright']['y']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img


def min_face_size_test(img_path):
    # img = cv2.imread(img_path)
    img = img_path
    counter = 1.0
    skipped = 0
    if img is not None:
        while True:
            imgcv = cv2.resize(img, (0, 0), fx=counter, fy=counter)
            print imgcv.shape,
            detections = facedemo.detect_largest(imgcv, upsamples=0)
            if detections:
                skipped = 0
                common.showImage(draw_faces(imgcv, [detections]))
                print("%s %s" % (detections['box']['bottomright']['x'] -
                                 detections['box']['topleft']['x'],
                                 detections['box']['bottomright']['y'] -
                                 detections['box']['topleft']['y']))
            else:
                skipped += 1
            counter -= 0.01
            if skipped >= 5:
                break

            key = 0xFF & cv2.waitKey(1)
            if key == 27:
                break


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    for i in range(10):
        _, img = cap.read()
    # min_face_size_test('/home/aestaq/Pictures/face.jpg')
    min_face_size_test(img)
