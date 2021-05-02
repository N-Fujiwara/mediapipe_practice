#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import cv2 as cv
import mediapipe as mp
import face_mesh
import pose
import hands

draw_color = (241, 252, 102)


class Holistic():
    @classmethod
    def set_args(cls, parser):
        parser.add_argument('--upper_body_only', action='store_true')
        parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.5)
        parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)
        parser.add_argument('--use_brect', action='store_true')
        return parser


    def __init__(self, args):
        self.upper_body_only_ = args.upper_body_only
        self.min_detection_confidence_ = args.min_detection_confidence
        self.min_tracking_confidence_ = args.min_tracking_confidence
        self.use_brect_ = args.use_brect

        mp_holistic = mp.solutions.holistic
        self.holistic_ = mp_holistic.Holistic(
            upper_body_only=self.upper_body_only_,
            min_detection_confidence=self.min_detection_confidence_,
            min_tracking_confidence=self.min_tracking_confidence_,
        )


    def process(self, process_img):
        results = self.holistic_.process(process_img)
        return results


    def draw(self, results, display_img):
        if results.face_landmarks is not None:
            # 外接矩形の計算
            brect = face_mesh.calc_bounding_rect(display_img, results.face_landmarks)
            # 描画
            display_img = face_mesh.draw_landmarks(display_img, results.face_landmarks)
            display_img = face_mesh.draw_bounding_rect(self.use_brect_, display_img, brect)

        if results.pose_landmarks is not None:
            # 外接矩形の計算
            brect = pose.calc_bounding_rect(display_img, results.pose_landmarks)
            # 描画
            display_img = pose.draw_landmarks(display_img, results.pose_landmarks,
                                         self.upper_body_only_)
            display_img = pose.draw_bounding_rect(self.use_brect_, display_img, brect)

        # Hands
        hand_results = [
            [results.left_hand_landmarks, 'L'],
            [results.right_hand_landmarks, 'R'],
            ]
        for hand_landmarks, handedness_str in hand_results:
            if hand_landmarks is not None:
                # 手の平重心計算
                cx, cy = hands.calc_palm_moment(display_img, hand_landmarks)
                # 外接矩形の計算
                brect = hands.calc_bounding_rect(display_img, hand_landmarks)
                # 描画
                display_img = hands.draw_landmarks(display_img, cx, cy,
                                                hand_landmarks,
                                                handedness_str,
                                                self.upper_body_only_)
                display_img = hands.draw_bounding_rect(self.use_brect_, display_img, brect)
        return display_img
