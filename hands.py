#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import cv2 as cv
import mediapipe as mp

draw_color = (241, 252, 102)


class Hands():
    @classmethod
    def set_args(cls, parser):
        parser.add_argument("--max_num_hands", type=int, default=2)
        parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
        parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)
        parser.add_argument('--use_brect', action='store_true')
        return parser


    def __init__(self, args):
        self.max_num_hands_ = args.max_num_hands
        self.min_detection_confidence_ = args.min_detection_confidence
        self.min_tracking_confidence_ = args.min_tracking_confidence
        self.use_brect_ = args.use_brect

        mp_hands = mp.solutions.hands
        self.hands_ = mp_hands.Hands(
            max_num_hands=self.max_num_hands_,
            min_detection_confidence=self.min_detection_confidence_,
            min_tracking_confidence=self.min_tracking_confidence_,
        )


    def process(self, process_img):
        results = self.hands_.process(process_img)
        return results


    def draw(self, results, display_img):
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # 手の平重心計算
                cx, cy = calc_palm_moment(display_img, hand_landmarks)
                # 外接矩形の計算
                brect = calc_bounding_rect(display_img, hand_landmarks)
                # 描画
                display_img = draw_landmarks(display_img, cx, cy,
                                             hand_landmarks, handedness)
                display_img = draw_bounding_rect(self.use_brect_, display_img, brect)
        return display_img


def calc_palm_moment(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    palm_array = np.empty((0, 2), int)

    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        if index == 0:  # 手首1
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 1:  # 手首2
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 5:  # 人差指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 9:  # 中指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 13:  # 薬指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 17:  # 小指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
    M = cv.moments(palm_array)
    cx, cy = 0, 0
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

    return cx, cy


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def draw_landmarks(image, cx, cy, landmarks, handedness):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # キーポイント
    for index, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue

        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append((landmark_x, landmark_y))

        if 0<=index and index<=20:
            cv.circle(image, (landmark_x, landmark_y), 5, draw_color, 2)
        if index in (4, 8, 12, 16, 20):  # 指先
            cv.circle(image, (landmark_x, landmark_y), 12, draw_color, 2)

        # if index == 0:  # 手首1
        # if index == 1:  # 手首2
        # if index == 2:  # 親指：付け根
        # if index == 3:  # 親指：第1関節
        # if index == 4:  # 親指：指先
        # if index == 5:  # 人差指：付け根
        # if index == 6:  # 人差指：第2関節
        # if index == 7:  # 人差指：第1関節
        # if index == 8:  # 人差指：指先
        # if index == 9:  # 中指：付け根
        # if index == 10:  # 中指：第2関節
        # if index == 11:  # 中指：第1関節
        # if index == 12:  # 中指：指先
        # if index == 13:  # 薬指：付け根
        # if index == 14:  # 薬指：第2関節
        # if index == 15:  # 薬指：第1関節
        # if index == 16:  # 薬指：指先
        # if index == 17:  # 小指：付け根
        # if index == 18:  # 小指：第2関節
        # if index == 19:  # 小指：第1関節
        # if index == 20:  # 小指：指先

    # 接続線
    if len(landmark_point) > 0:
        cont_points = [
            [2,3],   # 親指
            [5,7],   # 人差し指
            [9,11],  # 中指
            [13,15], # 薬指
            [17,19], # 小指
        ]
        for st, en in cont_points:
            for i in range(st, en+1):
                cv.line(image, landmark_point[i], landmark_point[i+1], draw_color, 2)

        # 手の平
        cv.line(image, landmark_point[0], landmark_point[1], draw_color, 2)
        cv.line(image, landmark_point[1], landmark_point[2], draw_color, 2)
        cv.line(image, landmark_point[2], landmark_point[5], draw_color, 2)
        cv.line(image, landmark_point[5], landmark_point[9], draw_color, 2)
        cv.line(image, landmark_point[9], landmark_point[13], draw_color, 2)
        cv.line(image, landmark_point[13], landmark_point[17], draw_color, 2)
        cv.line(image, landmark_point[17], landmark_point[0], draw_color, 2)

    # 重心 + 左右
    if len(landmark_point) > 0:
        # handedness.classification[0].index
        # handedness.classification[0].score
        cv.circle(image, (cx, cy), 12, draw_color, 2)
        cv.putText(image, handedness.classification[0].label[0],
                   (cx - 6, cy + 6), cv.FONT_HERSHEY_SIMPLEX, 0.6, draw_color,
                   2, cv.LINE_AA)  # label[0]:一文字目だけ = L or R
    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # 外接矩形
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     draw_color, 2)

    return image
