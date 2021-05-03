#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import cv2 as cv
import mediapipe as mp

draw_color = (241, 252, 102)


class FaceMesh():
    @classmethod
    def set_args(cls, parser):
        parser.add_argument("--max_num_faces",
                            help='max faces to detect',
                            type=int, default=1)
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
        self.max_num_faces_ = args.max_num_faces
        self.min_detection_confidence_ = args.min_detection_confidence
        self.min_tracking_confidence_ = args.min_tracking_confidence
        self.use_brect_ = args.use_brect

        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh_ = mp_face_mesh.FaceMesh(
            max_num_faces=self.max_num_faces_,
            min_detection_confidence=self.min_detection_confidence_,
            min_tracking_confidence=self.min_tracking_confidence_,
        )


    def process(self, process_img):
        results = self.face_mesh_.process(process_img)
        return results

    def draw(self, results, display_img):
        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                # 外接矩形の計算
                brect = calc_bounding_rect(display_img, face_landmarks)
                # 描画
                display_img = draw_landmarks(display_img, face_landmarks)
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


def draw_landmarks(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue

        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append((landmark_x, landmark_y))

        cv.circle(image, (landmark_x, landmark_y), 1, draw_color, 1)

    if len(landmark_point) > 0:
        # 参考：https://github.com/tensorflow/tfjs-models/blob/master/facemesh/mesh_map.jpg

        # 左眉毛(55：内側、46：外側)
        cv.line(image, landmark_point[55], landmark_point[65], draw_color, 2)
        cv.line(image, landmark_point[65], landmark_point[52], draw_color, 2)
        cv.line(image, landmark_point[52], landmark_point[53], draw_color, 2)
        cv.line(image, landmark_point[53], landmark_point[46], draw_color, 2)

        # 右眉毛(285：内側、276：外側)
        cv.line(image, landmark_point[285], landmark_point[295], draw_color,
                2)
        cv.line(image, landmark_point[295], landmark_point[282], draw_color,
                2)
        cv.line(image, landmark_point[282], landmark_point[283], draw_color,
                2)
        cv.line(image, landmark_point[283], landmark_point[276], draw_color,
                2)

        # 左目 (133：目頭、246：目尻)
        cv.line(image, landmark_point[133], landmark_point[173], draw_color,
                2)
        cv.line(image, landmark_point[173], landmark_point[157], draw_color,
                2)
        cv.line(image, landmark_point[157], landmark_point[158], draw_color,
                2)
        cv.line(image, landmark_point[158], landmark_point[159], draw_color,
                2)
        cv.line(image, landmark_point[159], landmark_point[160], draw_color,
                2)
        cv.line(image, landmark_point[160], landmark_point[161], draw_color,
                2)
        cv.line(image, landmark_point[161], landmark_point[246], draw_color,
                2)

        cv.line(image, landmark_point[246], landmark_point[163], draw_color,
                2)
        cv.line(image, landmark_point[163], landmark_point[144], draw_color,
                2)
        cv.line(image, landmark_point[144], landmark_point[145], draw_color,
                2)
        cv.line(image, landmark_point[145], landmark_point[153], draw_color,
                2)
        cv.line(image, landmark_point[153], landmark_point[154], draw_color,
                2)
        cv.line(image, landmark_point[154], landmark_point[155], draw_color,
                2)
        cv.line(image, landmark_point[155], landmark_point[133], draw_color,
                2)

        # 右目 (362：目頭、466：目尻)
        cv.line(image, landmark_point[362], landmark_point[398], draw_color,
                2)
        cv.line(image, landmark_point[398], landmark_point[384], draw_color,
                2)
        cv.line(image, landmark_point[384], landmark_point[385], draw_color,
                2)
        cv.line(image, landmark_point[385], landmark_point[386], draw_color,
                2)
        cv.line(image, landmark_point[386], landmark_point[387], draw_color,
                2)
        cv.line(image, landmark_point[387], landmark_point[388], draw_color,
                2)
        cv.line(image, landmark_point[388], landmark_point[466], draw_color,
                2)

        cv.line(image, landmark_point[466], landmark_point[390], draw_color,
                2)
        cv.line(image, landmark_point[390], landmark_point[373], draw_color,
                2)
        cv.line(image, landmark_point[373], landmark_point[374], draw_color,
                2)
        cv.line(image, landmark_point[374], landmark_point[380], draw_color,
                2)
        cv.line(image, landmark_point[380], landmark_point[381], draw_color,
                2)
        cv.line(image, landmark_point[381], landmark_point[382], draw_color,
                2)
        cv.line(image, landmark_point[382], landmark_point[362], draw_color,
                2)

        # 口 (308：右端、78：左端)
        cv.line(image, landmark_point[308], landmark_point[415], draw_color,
                2)
        cv.line(image, landmark_point[415], landmark_point[310], draw_color,
                2)
        cv.line(image, landmark_point[310], landmark_point[311], draw_color,
                2)
        cv.line(image, landmark_point[311], landmark_point[312], draw_color,
                2)
        cv.line(image, landmark_point[312], landmark_point[13], draw_color, 2)
        cv.line(image, landmark_point[13], landmark_point[82], draw_color, 2)
        cv.line(image, landmark_point[82], landmark_point[81], draw_color, 2)
        cv.line(image, landmark_point[81], landmark_point[80], draw_color, 2)
        cv.line(image, landmark_point[80], landmark_point[191], draw_color, 2)
        cv.line(image, landmark_point[191], landmark_point[78], draw_color, 2)

        cv.line(image, landmark_point[78], landmark_point[95], draw_color, 2)
        cv.line(image, landmark_point[95], landmark_point[88], draw_color, 2)
        cv.line(image, landmark_point[88], landmark_point[178], draw_color, 2)
        cv.line(image, landmark_point[178], landmark_point[87], draw_color, 2)
        cv.line(image, landmark_point[87], landmark_point[14], draw_color, 2)
        cv.line(image, landmark_point[14], landmark_point[317], draw_color, 2)
        cv.line(image, landmark_point[317], landmark_point[402], draw_color,
                2)
        cv.line(image, landmark_point[402], landmark_point[318], draw_color,
                2)
        cv.line(image, landmark_point[318], landmark_point[324], draw_color,
                2)
        cv.line(image, landmark_point[324], landmark_point[308], draw_color,
                2)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # 外接矩形
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     draw_color, 2)
    return image
