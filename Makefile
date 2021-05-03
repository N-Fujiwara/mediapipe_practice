.DEFAULT_GOAL := help

.PHONY: help
## help: prints this help message
help:
	@echo "Usage:"
	@sed -n 's/^##//p' ${MAKEFILE_LIST} | column -t -s ':' |  sed -e 's/^/ /'

.PHONY: easy-run
## easy-run: setup and run hands
easy-run: env-setup env-sync run-hands

.PHONY: env-setup
## env-setup: setup environment to develop
env-setup:
	pip install poetry

.PHONY: env-sync
## env-sync: sync environment to develop
env-sync:
	poetry install

.PHONY: env-update
## env-update: update latest environment to develop
env-update:
	pipenv update


TAR_DEVICE = 1  # my camera number
.PHONY: list-camera run-camera run-facedetection run-facemesh run-hands run-holistic run-objectron run-pose
## list-camera: list camera devices
list-camera:
	./camera_macos.py

## run-camera: test camera
run-camera:
	poetry run ./$@.py --device ${TAR_DEVICE}

## run-facedetection: face detection
run-facedetection:
	poetry run ./$@.py --device ${TAR_DEVICE}

## run-facemesh: face mesh
run-facemesh:
	poetry run ./$@.py --device ${TAR_DEVICE}

## run-hands: hands detection
run-hands:
	poetry run ./$@.py --device ${TAR_DEVICE}

## run-pose: pose detection
run-pose:
	poetry run ./$@.py --device ${TAR_DEVICE}

## run-objectron: 3d object detection
run-objectron:
	poetry run ./$@.py --device ${TAR_DEVICE}

## run-holistic: simultaneously 3 (face, hands, pose)
run-holistic:
	poetry run ./$@.py --device ${TAR_DEVICE}
