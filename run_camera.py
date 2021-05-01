#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import cv2 as cv
import mediapipe as mp
import os
from capture_device import CaptureDevice


def through(process_img, display_img):
    return display_img


def main():
    parser = argparse.ArgumentParser()
    parser = CaptureDevice.set_args(parser)
    args = parser.parse_args()

    cap = CaptureDevice(args)
    title = os.path.basename(__file__).split('.')[0]
    cap.loop(through, title)

if __name__ == '__main__':
    main()
