#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2 as cv
from collections import deque


def draw_texts(img, texts, font_scale=1.0, thickness=2):
    h, w, c = img.shape
    offset_x = 20  # 左下の座標
    initial_y = 0
    dy = int(img.shape[1] / 25)
    color = (241, 252, 102)  # RGB= 66FCF1
    texts = [texts] if type(texts) == str else texts
    for i, text in enumerate(texts):
        offset_y = initial_y + (i+1)*dy
        cv.putText(img, text, (offset_x, offset_y), cv.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv.LINE_AA)


def draw_result_on_img(img, texts, w_ratio=0.3, h_ratio=0.2, alpha=0.4):
    # 文字をのせるためのマットを作成する
    overlay = img.copy()
    pt1 = (0, 0)
    pt2 = (int(img.shape[1] * w_ratio), int(img.shape[0] * h_ratio))
    mat_color = (99, 98, 97)  # RGB C5C6C7
    fill = -1  # -1にすると塗りつぶし
    cv.rectangle(overlay, pt1, pt2, mat_color, fill)
    mat_img = cv.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    draw_texts(mat_img, texts)
    return mat_img


# original https://github.com/Kazuhito00/mediapipe-python-sample
class CvFpsCalc(object):
    def __init__(self, buffer_len=1):
        self._start_tick = cv.getTickCount()
        self._freq = 1000.0 / cv.getTickFrequency()
        self._difftimes = deque(maxlen=buffer_len)

    def get(self):
        current_tick = cv.getTickCount()
        different_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick
        self._difftimes.append(different_time)
        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        fps_rounded = round(fps, 2)
        return fps_rounded
