# mediapipe_practice

[mediapipe](https://google.github.io/mediapipe/) のpython solution動作サンプルです。

[Kazuhito00/mediapipe-python-sample](https://github.com/Kazuhito00/mediapipe-python-sample) を多いに参考にさせていただきました。

LICENSEは両者と同じく Apache-2.0 です。

## 実行環境

- python 3.8 or later
- poetry 1.1 or later

内部での主要ライブラリ： 下記 env-sync で入ります

- mediapipe 0.8.3 or later
- OpenCV 3.4.2 or later

動作確認

6年前の古いMacbook pro。GPU無し。

- macOS 11.2.3 (Big Sur)
- MacBook Pro (Retina, 13-inch, Early 2015)
- プロセッサ 2.7 GHz デュアルコアIntel Core i5
- メモリ 16 GB 1867 MHz DDR3
- グラフィックス Intel Iris Graphics 6100 1536 MB

## 実行

### TL;DR;

- `make easy-run` 一発で Hands認識 が動けばラッキー
- ダメなら下記手順を参考に修正ください
### 準備

- `make env-setup` poetryのインストール
- `make env-sync`  利用ライブラリのインストール

### カメラ接続確認

- `make run-camera` カメラ起動
- もししない場合は、 `Makefile` 中の `TAR_DEVICE` を別の値にしてください
- macOSなら `make list-camera` でデバイス一覧が出ます

### デモ動作

`make タスク名` で実行。タスク名一覧

```text
  run-camera          test camera
  run-facedetection   face detection
  run-facemesh        face mesh
  run-hands           hands detection
  run-pose            pose detection
  run-objectron       3d object detection
  run-holistic        simultaneously 3 (face, hands, pose)
```

### 引数詳細

タスクでは `poetry run ./タスク名.py オプション` が実行されるが、そのオプションは下記

| オプション名 | デフォルト値 | 意味 |
|--------------|-----------|------|
| --device | 0 | カメラデバイス番号 |
| --width | 960 | キャプチャ時の幅 |
| --height | 540 | キャプチャ時の高さ |
| --noflip | - | 左右フリップを止める＝外付けカメラモード |
| --use_brect | - | 指定すると検出矩形表示 |
| --upper_body_only | - | 上半身限定 |
| --min_detection_confidence | 0.7 or 0.5 | 検出信頼値の閾値 |
| --min_tracking_confidence | 0.5 or 0.99 | トラッキング信頼値の閾値 |
| --max_num_faces | 2 | 最大検出数 |
| --max_num_hands | 2 | 最大検出数 |
| --max_num_objects | 3 | 最大検出数 |
| --model_name | Cup | 検出対象 'Shoe', 'Chair', 'Cup', 'Camera'の4種類 |

eof
