#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import cv2 as cv
import mediapipe as mp

draw_color = (241, 252, 102)


class Pose():
    @classmethod
    def set_args(cls, parser):
        parser.add_argument('--upper_body_only', action='store_true')
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
        self.upper_body_only_ = args.upper_body_only
        self.min_detection_confidence_ = args.min_detection_confidence
        self.min_tracking_confidence_ = args.min_tracking_confidence
        self.use_brect_ = args.use_brect

        mp_pose = mp.solutions.pose
        self.pose_ = mp_pose.Pose(
            upper_body_only=self.upper_body_only_,
            min_detection_confidence=self.min_detection_confidence_,
            min_tracking_confidence=self.min_tracking_confidence_,
        )


    def process(self, process_img):
        results = self.pose_.process(process_img)
        return results


    def draw(self, results, display_img):
        if results.pose_landmarks is not None:
            # 外接矩形の計算
            brect = calc_bounding_rect(display_img, results.pose_landmarks)
            # 描画
            display_img = draw_landmarks(display_img, results.pose_landmarks,
                                         self.upper_body_only_)
            display_img = draw_bounding_rect(self.use_brect_, display_img, brect)
        return display_img


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


def draw_landmarks(image, landmarks, upper_body_only, visibility_th=0.5):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z
        landmark_point.append([landmark.visibility, (landmark_x, landmark_y)])

        if landmark.visibility < visibility_th:
            continue

        if 0<=index and index<=32:
            cv.circle(image, (landmark_x, landmark_y), 5, draw_color, 2)

        # if index == 0:  # 鼻
        # if index == 1:  # 右目：目頭
        # if index == 2:  # 右目：瞳
        # if index == 3:  # 右目：目尻
        # if index == 4:  # 左目：目頭
        # if index == 5:  # 左目：瞳
        # if index == 6:  # 左目：目尻
        # if index == 7:  # 右耳
        # if index == 8:  # 左耳
        # if index == 9:  # 口：左端
        # if index == 10:  # 口：左端
        # if index == 11:  # 右肩
        # if index == 12:  # 左肩
        # if index == 13:  # 右肘
        # if index == 14:  # 左肘
        # if index == 15:  # 右手首
        # if index == 16:  # 左手首
        # if index == 17:  # 右手1(外側端)
        # if index == 18:  # 左手1(外側端)
        # if index == 19:  # 右手2(先端)
        # if index == 20:  # 左手2(先端)
        # if index == 21:  # 右手3(内側端)
        # if index == 22:  # 左手3(内側端)
        # if index == 23:  # 腰(右側)
        # if index == 24:  # 腰(左側)
        # if index == 25:  # 右ひざ
        # if index == 26:  # 左ひざ
        # if index == 27:  # 右足首
        # if index == 28:  # 左足首
        # if index == 29:  # 右かかと
        # if index == 30:  # 左かかと
        # if index == 31:  # 右つま先
        # if index == 32:  # 左つま先

        if not upper_body_only:
            cv.putText(image, "z:" + str(round(landmark_z, 3)),
                       (landmark_x - 10, landmark_y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 1,
                       cv.LINE_AA)

    if len(landmark_point) > 0:
        # 右目
        if landmark_point[1][0] > visibility_th and landmark_point[2][
                0] > visibility_th:
            cv.line(image, landmark_point[1][1], landmark_point[2][1],
                    draw_color, 2)
        if landmark_point[2][0] > visibility_th and landmark_point[3][
                0] > visibility_th:
            cv.line(image, landmark_point[2][1], landmark_point[3][1],
                    draw_color, 2)

        # 左目
        if landmark_point[4][0] > visibility_th and landmark_point[5][
                0] > visibility_th:
            cv.line(image, landmark_point[4][1], landmark_point[5][1],
                    draw_color, 2)
        if landmark_point[5][0] > visibility_th and landmark_point[6][
                0] > visibility_th:
            cv.line(image, landmark_point[5][1], landmark_point[6][1],
                    draw_color, 2)

        # 口
        if landmark_point[9][0] > visibility_th and landmark_point[10][
                0] > visibility_th:
            cv.line(image, landmark_point[9][1], landmark_point[10][1],
                    draw_color, 2)

        # 肩
        if landmark_point[11][0] > visibility_th and landmark_point[12][
                0] > visibility_th:
            cv.line(image, landmark_point[11][1], landmark_point[12][1],
                    draw_color, 2)

        # 右腕
        if landmark_point[11][0] > visibility_th and landmark_point[13][
                0] > visibility_th:
            cv.line(image, landmark_point[11][1], landmark_point[13][1],
                    draw_color, 2)
        if landmark_point[13][0] > visibility_th and landmark_point[15][
                0] > visibility_th:
            cv.line(image, landmark_point[13][1], landmark_point[15][1],
                    draw_color, 2)

        # 左腕
        if landmark_point[12][0] > visibility_th and landmark_point[14][
                0] > visibility_th:
            cv.line(image, landmark_point[12][1], landmark_point[14][1],
                    draw_color, 2)
        if landmark_point[14][0] > visibility_th and landmark_point[16][
                0] > visibility_th:
            cv.line(image, landmark_point[14][1], landmark_point[16][1],
                    draw_color, 2)

        # 右手
        if landmark_point[15][0] > visibility_th and landmark_point[17][
                0] > visibility_th:
            cv.line(image, landmark_point[15][1], landmark_point[17][1],
                    draw_color, 2)
        if landmark_point[17][0] > visibility_th and landmark_point[19][
                0] > visibility_th:
            cv.line(image, landmark_point[17][1], landmark_point[19][1],
                    draw_color, 2)
        if landmark_point[19][0] > visibility_th and landmark_point[21][
                0] > visibility_th:
            cv.line(image, landmark_point[19][1], landmark_point[21][1],
                    draw_color, 2)
        if landmark_point[21][0] > visibility_th and landmark_point[15][
                0] > visibility_th:
            cv.line(image, landmark_point[21][1], landmark_point[15][1],
                    draw_color, 2)

        # 左手
        if landmark_point[16][0] > visibility_th and landmark_point[18][
                0] > visibility_th:
            cv.line(image, landmark_point[16][1], landmark_point[18][1],
                    draw_color, 2)
        if landmark_point[18][0] > visibility_th and landmark_point[20][
                0] > visibility_th:
            cv.line(image, landmark_point[18][1], landmark_point[20][1],
                    draw_color, 2)
        if landmark_point[20][0] > visibility_th and landmark_point[22][
                0] > visibility_th:
            cv.line(image, landmark_point[20][1], landmark_point[22][1],
                    draw_color, 2)
        if landmark_point[22][0] > visibility_th and landmark_point[16][
                0] > visibility_th:
            cv.line(image, landmark_point[22][1], landmark_point[16][1],
                    draw_color, 2)

        # 胴体
        if landmark_point[11][0] > visibility_th and landmark_point[23][
                0] > visibility_th:
            cv.line(image, landmark_point[11][1], landmark_point[23][1],
                    draw_color, 2)
        if landmark_point[12][0] > visibility_th and landmark_point[24][
                0] > visibility_th:
            cv.line(image, landmark_point[12][1], landmark_point[24][1],
                    draw_color, 2)
        if landmark_point[23][0] > visibility_th and landmark_point[24][
                0] > visibility_th:
            cv.line(image, landmark_point[23][1], landmark_point[24][1],
                    draw_color, 2)

        if len(landmark_point) > 25:
            # 右足
            if landmark_point[23][0] > visibility_th and landmark_point[25][
                    0] > visibility_th:
                cv.line(image, landmark_point[23][1], landmark_point[25][1],
                        draw_color, 2)
            if landmark_point[25][0] > visibility_th and landmark_point[27][
                    0] > visibility_th:
                cv.line(image, landmark_point[25][1], landmark_point[27][1],
                        draw_color, 2)
            if landmark_point[27][0] > visibility_th and landmark_point[29][
                    0] > visibility_th:
                cv.line(image, landmark_point[27][1], landmark_point[29][1],
                        draw_color, 2)
            if landmark_point[29][0] > visibility_th and landmark_point[31][
                    0] > visibility_th:
                cv.line(image, landmark_point[29][1], landmark_point[31][1],
                        draw_color, 2)

            # 左足
            if landmark_point[24][0] > visibility_th and landmark_point[26][
                    0] > visibility_th:
                cv.line(image, landmark_point[24][1], landmark_point[26][1],
                        draw_color, 2)
            if landmark_point[26][0] > visibility_th and landmark_point[28][
                    0] > visibility_th:
                cv.line(image, landmark_point[26][1], landmark_point[28][1],
                        draw_color, 2)
            if landmark_point[28][0] > visibility_th and landmark_point[30][
                    0] > visibility_th:
                cv.line(image, landmark_point[28][1], landmark_point[30][1],
                        draw_color, 2)
            if landmark_point[30][0] > visibility_th and landmark_point[32][
                    0] > visibility_th:
                cv.line(image, landmark_point[30][1], landmark_point[32][1],
                        draw_color, 2)
    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # 外接矩形
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     draw_color, 2)

    return image
