#!/usr/bin/env python

from face import Face
import cv2
import common

# url = 'rtsp://admin:admin@192.168.0.100/live'
# url = '/home/aestaq/Videos/qb.mp4'
url = "/home/aestaq/Videos/test.avi"
# url = 0

DETECTOR = 'dlib'
TRACKING = False

if TRACKING:
    from multitracker import MultiTracker, CorrelationTracker
    removalConfig = {
        'invisible_count': 35,
        'overlap_thresh': 0.9,
        'overlap_invisible_count': 5,
        'corner_percentage': 0.1,
        'corner_invisible_count': 5
        }
    mtracker = MultiTracker(SingleTrackerType=CorrelationTracker)
    # mtracker = MultiTracker(SingleTrackerType=CorrelationTracker,
    #                         removalConfig=removalConfig)
    # mtracker = MultiTracker(SingleTrackerType = cv2.TrackerKCF_create)


def to_bbox(detections):
    return [{
            "class": 'face',
            "prob": 0.99,
            "box": {
                "topleft": {'x': x, 'y': y},
                "bottomright": {'x': (x+w), 'y': (y+h)}
                },
            } for (x, y, w, h) in detections]


def demo_video(video_file):
    import time
    facedemo = Face(detector_method=DETECTOR, recognition_method=None)
    cap = common.VideoStream(video_file, queueSize=4).start()
    time.sleep(1)
    total_t, counter = 0, 0
    t = common.clock()

    while not cap.stopped:
        imgcv = cap.read()
        if imgcv is not None:
            counter += 1
            detections = facedemo.detect(imgcv, upsamples=0)
            ids = range(len(detections))

            # temp = mtracker.update(imgcv, to_cvbox(detections))
            # cvboxes, ids = [], []
            # for tid,tracker in mtracker.trackers.items():
            #     if tracker.visible_count > 3 and tracker.consecutive_invisible_count<10:
            #         cvboxes.append(tracker.bbox)
            #         ids.append(tid)
            # detections = to_bbox(cvboxes)

            print(detections)
            common.showImage(common.drawObjects(imgcv, detections, ids))

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        t1 = common.clock()
        dt = t1-t
        t = t1
        total_t += dt
        print(counter/total_t)


if __name__ == '__main__':
    demo_video(url)
