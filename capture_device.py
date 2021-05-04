#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import copy
import cv2 as cv
import cv_util

class CaptureDevice:
    @classmethod
    def set_args(cls, parser):
        parser.add_argument("--device", type=int, default=0)
        parser.add_argument("--width", help='cap width', type=int, default=960)
        parser.add_argument("--height", help='cap height', type=int, default=540)
        parser.add_argument("--noflip", help='flip capture', action='store_true')
        return parser

    def __init__(self, args):
        cap = cv.VideoCapture(args.device)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
        self.cap_ = cap
        self.args_ = args
        self.cvFpsCalc_ = cv_util.CvFpsCalc(buffer_len=10)


    def capture(self):
        ret, image = self.cap_.read()
        if not ret:
            return None

        if not self.args_.noflip:
            image = cv.flip(image, 1)  # ミラー表示
        return image


    def release(self):
        self.cap_.release()


    def loop(self, proc_func, title):
        while True:
            image = self.capture()
            display_img = copy.deepcopy(image)
            process_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            display_img = proc_func(process_img, display_img)

            display_fps = self.cvFpsCalc_.get()
            text_fps = [title, "FPS:" + str(display_fps)]
            display_img = cv_util.draw_result_on_img(display_img, text_fps)

            key = cv.waitKey(1)  # キー処理(ESC：終了) 
            if key == 27:  # ESC
                break
            cv.imshow(title, display_img)

        self.cap_.release()
        cv.destroyAllWindows()
