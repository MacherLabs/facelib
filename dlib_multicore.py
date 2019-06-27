#!/usr/bin/env python

# from collections import deque
import cv2
import common
from face import Face

url = "/home/aestaq/Videos/test.avi"
# url = 0
DETECTOR = 'dlib'
facedemo = Face(detector_method=DETECTOR, recognition_method=None)


def process_frame(frame):
    detections = facedemo.detect(frame, upsamples=0)
    # frame = draw_faces(frame, detections)
    return frame, detections


if __name__ == '__main__':
    # threadn = cv2.getNumberOfCPUs()
    # print threadn
    # from multiprocessing import Pool
    # pool = Pool(processes = threadn)
    # pending = deque()
    multi = common.Multicore(process_frame)

    total_t, counter = 0, 0
    cap = common.VideoStream(url, queueSize=4).start()
    t = common.clock()

    while not cap.stopped:
        # while len(pending) > 0 and pending[0].ready():
        #     frame = pending.popleft().get()
        #     common.showImage(frame)

        # if len(pending) < threadn:
        #     imgcv = cap.read()
        #     if imgcv is not None:
        #         counter += 1
        #         task = pool.apply_async(process_frame, (imgcv.copy(),))
        #         pending.append(task)
        #     else:
        #         print "Cannot read frame"
        #         break

        imgcv = cap.read()
        print imgcv.shape,
        if imgcv is not None:
            counter += 1
            out = multi.run(imgcv)
            if out is not None:
                frame, detections = out
                common.showImage(common.drawObjects(frame, detections))

        t1 = common.clock()
        dt = t1-t
        t = t1
        total_t += dt
        print counter/total_t

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
