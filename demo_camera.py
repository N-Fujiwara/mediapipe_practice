#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse
import cv2 as cv
import numpy as np
import mediapipe as mp
from capture_device import CaptureDevice


def through(process_img, display_img):
    return display_img


def main():
    parser = argparse.ArgumentParser()
    parser = CaptureDevice.set_args(parser)
    args = parser.parse_args()

    cap = CaptureDevice(args)

    while True:
        image = cap.capture()
        display_img = copy.deepcopy(image)
        process_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        display_img = through(process_img, display_img)

        key = cv.waitKey(1)  # キー処理(ESC：終了) 
        if key == 27:  # ESC
            break
        cv.imshow('MediaPipe Face Detection Demo', display_img)

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
