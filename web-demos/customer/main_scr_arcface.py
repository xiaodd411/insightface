#!/usr/bin/env python

import os
import os.path as osp
import argparse
import cv2
import numpy as np
import onnxruntime
from scrfd import SCRFD
from arcface_onnx import ArcFaceONNX
from typing import List

onnxruntime.set_default_logger_severity(3)

assets_dir = osp.expanduser('~/.insightface/models/buffalo_l')

# Initialize the SCRFD detector and ArcFace recognition model
detector = SCRFD(os.path.join(assets_dir, 'det_10g.onnx'))
detector.prepare(0)

model_path = os.path.join(assets_dir, 'w600k_r50.onnx')
rec = ArcFaceONNX(model_path)
rec.prepare(0)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, choices=['detection'], help="Action to perform: 'detection' for face detection")
    parser.add_argument('input_path', type=str, help="Image file or directory for processing")
    return parser.parse_args()

def process_image(image_path: str):
    image = cv2.imread(image_path)

    # Perform face detection using SCRFD
    bboxes, kpss = detector.autodetect(image, max_num=1)

    if bboxes.shape[0] == 0:
        return image_path, False, None, None, "Face not found"

    # Extract confidence score for the detected face(s)
    face_score = bboxes[0][4]  # Assuming the confidence score is in the 5th column of the bboxes array

    kps = kpss[0]
    feat = rec.get(image, kps)

    return image_path, True, face_score, feat, "Face found"

def process_directory(directory_path: str) -> List[tuple]:
    image_files = [osp.join(directory_path, f) for f in os.listdir(directory_path) if f.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif'))]
    return list(map(process_image, image_files))

def func(args):
    action = args.action
    input_path = args.input_path

    if action == 'detection':
        if osp.isdir(input_path):
            results = process_directory(input_path)
            for result in results:
                image_path, face_found, score, feature, message = result
                # TODO 向量信息待输出,同时根据向量计算相似度的也待实现
                print(f"Image: {image_path}, Face Found: {face_found}, Face Score: {score:.4f}, Message: {message}")
        elif osp.isfile(input_path):
            result = process_image(input_path)
            image_path, face_found, score, feature, message = result
            # TODO 向量信息待输出，这里得到的向量是否是归一化的待确认？,同时根据向量计算相似度的也待实现
            print(f"Image: {image_path}, Face Found: {face_found}, Face Score: {score:.4f}, Message: {message}")
        else:
            print("Invalid input path. Please provide a valid image file or directory.")
    else:
        print("Invalid action. Please specify 'detection'.")

if __name__ == '__main__':
    args = parse_args()
    func(args)
