#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import os
from capture_device import CaptureDevice
from face_mesh import FaceMesh as Model


def main():
    parser = argparse.ArgumentParser()
    parser = CaptureDevice.set_args(parser)
    parser = Model.set_args(parser)
    args = parser.parse_args()

    cap = CaptureDevice(args)
    model = Model(args)
    title = os.path.basename(__file__).split('.')[0]
    cap.loop(model.process, title)

if __name__ == '__main__':
    main()
