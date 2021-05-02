#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import os
from capture_device import CaptureDevice
from face_detection import FaceDetection as Model


def main():
    parser = argparse.ArgumentParser()
    parser = CaptureDevice.set_args(parser)
    parser = Model.set_args(parser)
    args = parser.parse_args()

    model = Model(args)
    def _run(process_img, display_img):
        results = model.process(process_img)
        display_img = model.draw(results, display_img)
        return display_img

    title = os.path.basename(__file__).split('.')[0]
    cap = CaptureDevice(args)
    cap.loop(_run, title)

if __name__ == '__main__':
    main()
