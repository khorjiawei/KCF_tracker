import KCF
import numpy as np
import os as os
import cv2 as cv2
import utils

dataset_path = "datasets/bolt1"
# dataset_path = "datasets/David/img"

if os.path.isdir(dataset_path):
    dataset_files = os.listdir(dataset_path)
    dataset_files.sort()

    # Construct full path to each image
    for i, f in enumerate(dataset_files):
        dataset_files[i] = os.path.join(dataset_path, f)

tracker = KCF.KCF()
for i, f in enumerate(dataset_files):
    im = cv2.imread(dataset_files[i], cv2.IMREAD_GRAYSCALE)
    if (i == 0):
        roi_rect = cv2.selectROI(im)
        if len(roi_rect) == 0:
            raise Exception("No bounding box selected")
        tracker.init(im, roi_rect)
    else:
        tracker.update(im)
        print(tracker.bbox_.center())

    # Plot and visualise track result
    bbox = tracker.bbox_
    bbox = (bbox.top_left_x, bbox.top_left_y, bbox.width, bbox.height)
    cv2.rectangle(im, (bbox[0], bbox[1]),
                  (bbox[0]+bbox[2], bbox[1]+bbox[3]), (255, 255, 0))
    cv2.imshow("im", im)
    usr_key = cv2.waitKey(100)
    if (usr_key == ord('q')):
        break

cv2.waitKey()
