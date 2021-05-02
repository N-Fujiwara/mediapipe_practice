#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import cv2 as cv
import mediapipe as mp

draw_color = (241, 252, 102)


class Objectron():
    @classmethod
    def set_args(cls, parser):
        parser.add_argument('--static_image_mode', action='store_true',
                            help='static image = no tracking')
        parser.add_argument("--max_num_objects",
                            help='max_num_objects',
                            type=int,
                            default=5)
        parser.add_argument("--min_detection_confidence",
                            help='min_detection_confidence',
                            type=float,
                            default=0.5)
        parser.add_argument("--min_tracking_confidence",
                            help='min_tracking_confidence',
                            type=int,
                            default=0.99)
        parser.add_argument("--model_name",
                            help="{'Shoe', 'Chair', 'Cup', 'Camera'}",
                            type=str,
                            default='Cup')
        parser.add_argument('--verbose', action='store_true')
        return parser


    def __init__(self, args):
        self.static_image_mode_ = args.static_image_mode
        self.max_num_objects_ = args.max_num_objects
        self.min_detection_confidence_ = args.min_detection_confidence
        self.min_tracking_confidence_ = args.min_tracking_confidence
        self.model_name_ = args.model_name
        self.verbose_ = args.verbose

        self.mp_objectron_ = mp.solutions.objectron
        self.objectron_ = self.mp_objectron_.Objectron(
            static_image_mode=self.static_image_mode_,
            max_num_objects=self.max_num_objects_,
            min_detection_confidence=self.min_detection_confidence_,
            min_tracking_confidence=self.min_tracking_confidence_,
            model_name=self.model_name_,
        )
        self.mp_drawing_ = mp.solutions.drawing_utils


    def process(self, process_img):
        results = self.objectron_.process(process_img)
        return results


    def draw(self, results, display_img):
        if results.detected_objects is not None:
            for detected_object in results.detected_objects:
                self.mp_drawing_.draw_landmarks(display_img,
                                          detected_object.landmarks_2d,
                                          self.mp_objectron_.BOX_CONNECTIONS)
                self.mp_drawing_.draw_axis(display_img, detected_object.rotation,
                                           detected_object.translation)
                # キーポイント確認用
                draw_landmarks(display_img, detected_object.landmarks_2d)
        return display_img


def draw_landmarks(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([(landmark_x, landmark_y)])

        if index == 0:  # 重心
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 1:  #
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 2:  #
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 3:  #
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 4:  #
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 5:  #
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 6:  #
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 7:  #
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 8:  #
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # 外接矩形
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 255, 0), 2)

    return image
