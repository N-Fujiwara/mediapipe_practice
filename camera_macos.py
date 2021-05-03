#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import subprocess


def listup():
    # カメラリストの取得
    cmd = 'system_profiler SPCameraDataType | grep "^    [^ ]" | sed "s/    //" | sed "s/://" '
    res = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
    # 出力結果の加工
    ret = res.stdout.decode('utf-8')
    camera_list = list(filter(lambda a: a != "", ret.split('\n')))
    return camera_list


def search_id(camera_name):
    camera_list = listup()
    # 指定カメラの取得
    for index, row in enumerate(camera_list):
        if row.find(camera_name) != -1: 
          return index
    raise Exception('指定カメラなし')


if __name__ == '__main__':
    camera_list = listup()
    print('# device-index, camera_name')
    for i, camera in enumerate(camera_list):
        print('  ', i, camera)
