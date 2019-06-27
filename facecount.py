#!/usr/bin/env python

from face import Face
import common
import cv2
from multitracker import MultiTracker, CorrelationTracker

# url = 'rtsp://admin:admin@192.168.0.100/live'
url = "/home/aestaq/Videos/test.avi"
# url = 1

DETECTOR = 'dlib'
LINE = {'x1': 640, 'y1': 0, 'x2': 640, 'y2': 720}

removalConfig = {
    'invisible_count': 35,
    'overlap_thresh': 0.9,
    'overlap_invisible_count': 5,
    'corner_percentage': 0.1,
    'corner_invisible_count': 5}


def to_bbox(detections):
    return [{
            "class": 'face',
            "prob": 0.99,
            "box": {
                "topleft": {'x': x, 'y': y},
                "bottomright": {'x': (x+w), 'y': (y+h)}
                },
            } for (x, y, w, h) in detections]


def get_pos(bbox):
    x, y, w, h = bbox
    cx, cy = x+w/2, y+h/2   # centroid coordinates
    side = 'Positive' if (cx-LINE['x1'])*(LINE['y2']-LINE['y1']) - (
            cy-LINE['y1'])*(LINE['x2']-LINE['x1']) >= 0 else 'Negative'
    return side


def demo_video(video_file):
    facedemo = Face(detector_method=DETECTOR, recognition_method=None)
    mtracker = MultiTracker(SingleTrackerType=CorrelationTracker)
    # mtracker = MultiTracker(SingleTrackerType=CorrelationTracker,
    #                         removalConfig=removalConfig)
    # mtracker = MultiTracker(SingleTrackerType = cv2.TrackerKCF_create)

    cap = common.VideoStream(video_file, queueSize=4).start()
    cv2.waitKey(500)
    Outcount, Incount = 0, 0

    while not cap.stopped:
        t = common.clock()
        total_t, counter = 0, 0

        imgcv = cap.read()
        if imgcv is not None:
            counter += 1
            detections = facedemo.detect(imgcv, upsamples=0)
            mtracker.update(imgcv, common.toCvbox(detections))
            cvboxes, ids = [], []

            for tid, tracker in mtracker.trackers.items():
                if tracker.visible_count > 3 and tracker.consecutive_invisible_count < 10:
                    state_current = get_pos(tracker.bbox)
                    try:
                        if state_current != tracker.regionside:
                            tracker.statechange += 1
                            print state_current, tracker.regionside, tracker.statechange
                            if state_current == 'Positive':
                                if tracker.statechange % 2:
                                    Incount += 1
                                else:
                                    Outcount -= 1
                            else:
                                if tracker.statechange % 2:
                                    Outcount += 1
                                else:
                                    Incount -= 1
                            tracker.regionside = state_current
                    except AttributeError:
                        tracker.regionside = state_current
                        tracker.statechange = 0

                    cvboxes.append(tracker.bbox)
                    ids.append(tid)

            detections = to_bbox(cvboxes)
            print Incount, Outcount
            cv2.line(imgcv, (LINE['x1'], LINE['y1']), (LINE['x2'], LINE['y2']),
                     (0, 0, 255), 4)
            imgcv = common.drawLabel(imgcv, "IN:%d  OUT:%d" % (Incount, Outcount),
                                     (10, 10), color=(0, 0, 255))
            common.showImage(common.drawObjects(imgcv, detections, ids))

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        t1 = common.clock()
        dt = t1-t
        t = t1
        total_t += dt
        print counter/total_t


if __name__ == '__main__':
    demo_video(url)
