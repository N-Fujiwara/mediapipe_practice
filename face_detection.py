#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import cv2 as cv
import mediapipe as mp

draw_color = (241, 252, 102)


class FaceDetection():
    @classmethod
    def set_args(cls, parser):
        parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
        parser.add_argument('--verbose', action='store_true')
        return parser


    def __init__(self, args):
        self.min_detection_confidence_ = args.min_detection_confidence
        self.verbose_ = args.verbose

        mp_face_detection = mp.solutions.face_detection
        self.face_detection_ = mp_face_detection.FaceDetection(
            min_detection_confidence=self.min_detection_confidence_,
        )


    def process(self, process_img):
        results = self.face_detection_.process(process_img)
        return results


    def draw(self, results, display_img):
        if results.detections is not None:
            for detection in results.detections:
                # 描画
                display_img = draw_detection(display_img, detection, self.verbose_)
        return display_img


def draw_detection(image, detection, verbose):
    image_width, image_height = image.shape[1], image.shape[0]

    if verbose:
        print(detection)
        for i in range(6):
            print(detection.location_data.relative_keypoints[i])

    # バウンディングボックス
    bbox = detection.location_data.relative_bounding_box
    bbox.xmin = int(bbox.xmin * image_width)
    bbox.ymin = int(bbox.ymin * image_height)
    bbox.width = int(bbox.width * image_width)
    bbox.height = int(bbox.height * image_height)

    cv.rectangle(image, (int(bbox.xmin), int(bbox.ymin)),
                 (int(bbox.xmin + bbox.width), int(bbox.ymin + bbox.height)),
                 draw_color, 2)

    # スコア・ラベルID
    text = f'Id={detection.label_id[0]}  score={round(detection.score[0], 3)}'
    cv.putText(
        image, text,
        (int(bbox.xmin), int(bbox.ymin) - 20), cv.FONT_HERSHEY_SIMPLEX, 1.0,
        draw_color, 2, cv.LINE_AA)

    # 描画: 0->5 ＝ 右目, 左目, 鼻, 口, 右耳、左耳
    for i in range(6):
        keypoint = detection.location_data.relative_keypoints[i]
        keypoint.x = int(keypoint.x * image_width)
        keypoint.y = int(keypoint.y * image_height)
        cv.circle(image, (int(keypoint.x), int(keypoint.y)), 5, draw_color, 2)

    return image
