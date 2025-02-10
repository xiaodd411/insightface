#!/usr/bin/env python

import os
import os.path as osp
import argparse
import cv2
import numpy as np
import onnxruntime
import json
from scrfd import SCRFD
from arcface_onnx import ArcFaceONNX
from typing import List, Dict, Any
from numpy.linalg import norm

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
    parser.add_argument('action', type=str, choices=['detection'],
                        help="Action to perform: 'detection' for face detection")
    parser.add_argument('input_path', type=str, help="Image file or directory for processing")
    return parser.parse_args()


def process_image(image_path: str) -> Dict[str, Any]:
    res = []
    try:
        image = cv2.imread(image_path)
        # 使用SCRFD执行面部检测
        bboxes, kpss = detector.autodetect(image, max_num=10)

        if bboxes.shape[0] == 0:
            res.append({
                "face_found": False,
                "score": None,
                "feature": None,
                "message": "未检测到人脸"
            })
            return {image_path: res}

        for i, bbox in enumerate(bboxes):
            # 提取检测到的面孔的置信度得分
            score = bbox[4]  # 置信度得分在Bboxes数组的第五列中
            kps = kpss[i]
            feat = rec.get(image, kps)
            # TODO 这里的特征向量非归一化的，需要归一化
            res.append({
                "face_found": True,
                "score": float(score),
                "feature": (feat / norm(feat)).tolist(),
                # "message": "Face found"
            })
        return {image_path: res}
    except Exception as e:
        res.append({"face_found": False, "score": None, "feature": None, "message": "处理图片时出错:" + str(e)})
        return {image_path: res}


def process_directory(directory_path: str) -> Dict[str, Any]:
    image_files = [osp.join(directory_path, f) for f in os.listdir(directory_path) if
                   f.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif'))]
    results = {}
    for image_file in image_files:
        result = process_image(image_file)
        results.update(result)
    return results


def func(args) -> str:
    action = args.action
    input_path = args.input_path

    try:
        if action == 'detection':
            if osp.isdir(input_path):
                results = process_directory(input_path)
            elif osp.isfile(input_path):
                results = process_image(input_path)
            else:
                return json.dumps({})
            return json.dumps(results)
        else:
            return json.dumps({})
    except Exception as e:
        return json.dumps({})


if __name__ == '__main__':

    args = parse_args()
    print(func(args))
