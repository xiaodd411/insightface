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
    parser.add_argument('action', type=str, choices=['detection', 'detectionAndProcess'],
                        help="Action to perform: 'detection' for face detection, 'detectionAndProcess' for detection and cropping")
    parser.add_argument('input_path', type=str, help="Image file or directory for processing")
    return parser.parse_args()


def expand_bbox(bbox, scale=1.2, img_width=None, img_height=None):
    """扩大人脸框并确保不会超出图像边界"""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    # 增加边界
    new_width = int(width * scale)
    new_height = int(height * scale)

    # 重新计算边界
    new_x1 = max(x1 - (new_width - width) // 2, 0)
    new_y1 = max(y1 - (new_height - height) // 2, 0)
    new_x2 = min(new_x1 + new_width, img_width)
    new_y2 = min(new_y1 + new_height, img_height)

    return [new_x1, new_y1, new_x2, new_y2]


def save_cropped_face(img, bbox, image_name, output_dir, face_index):
    """裁剪人脸并保存到指定目录"""
    x1, y1, x2, y2, score = bbox.astype(int)

    # 获取图像尺寸
    img_height, img_width = img.shape[:2]

    # 扩展边界，增加 20% 的边界
    expanded_bbox = expand_bbox([x1, y1, x2, y2], scale=1.8, img_width=img_width, img_height=img_height)

    # 裁剪人脸部分
    cropped_face = img[expanded_bbox[1]:expanded_bbox[3], expanded_bbox[0]:expanded_bbox[2]]

    # 创建文件夹以存储裁剪的图片
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 使用人脸顺序命名裁剪图片
    cropped_image_path = os.path.join(output_dir, f"{face_index + 1}.jpg")
    cv2.imwrite(cropped_image_path, cropped_face)  # 保存裁剪图片
    return cropped_image_path


def process_image(image_path: str, process_faces=False) -> Dict[str, Any]:
    res = []
    # 获取图片文件名称(不包含后缀)
    image_name = osp.splitext(osp.basename(image_path))[0]
    try:
        image = cv2.imread(image_path)
        # 使用SCRFD执行面部检测
        bboxes, kpss = detector.autodetect(image, max_num=10)

        if bboxes.shape[0] == 0:
            res.append({
                "found": False,
                "score": None,
                "feature": None,
                "message": "未检测到人脸"
            })
            return {image_name: res}

        # 创建一个存储裁剪图像的文件夹
        output_dir = osp.join(osp.dirname(image_path), image_name)
        cropped_paths = []

        for i, bbox in enumerate(bboxes):
            score = bbox[4]  # 置信度得分在Bboxes数组的第五列中
            kps = kpss[i]
            feat = rec.get(image, kps)

            # 归一化特征向量
            feat = feat / norm(feat)

            result_entry = {
                "found": True,
                "score": float(score),
                "feature": feat.tolist(),
                "originName": image_name,
            }

            if process_faces:
                # 如果需要裁剪人脸，调用裁剪函数
                cropped_image_path = save_cropped_face(image, bbox, image_name, output_dir, i)
                result_entry["path"] = cropped_image_path
                cropped_paths.append(cropped_image_path)

            res.append(result_entry)

        return {image_name: res}

    except Exception as e:
        res.append({"found": False, "score": None, "feature": None, "message": "处理图片时出错:" + str(e)})
        return {image_name: res}


def process_directory(directory_path: str, process_faces=False) -> Dict[str, Any]:
    image_files = [osp.join(directory_path, f) for f in os.listdir(directory_path) if
                   f.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif'))]
    results = {}
    for image_file in image_files:
        result = process_image(image_file, process_faces)
        results.update(result)
    return results


def func(args) -> str:
    action = args.action
    input_path = args.input_path

    try:
        if action == 'detection':
            if osp.isdir(input_path):
                results = process_directory(input_path, process_faces=False)
            elif osp.isfile(input_path):
                results = process_image(input_path, process_faces=False)
            else:
                return json.dumps({})
            return json.dumps(results)

        elif action == 'detectionAndProcess':
            if osp.isdir(input_path):
                results = process_directory(input_path, process_faces=True)
            elif osp.isfile(input_path):
                results = process_image(input_path, process_faces=True)
            else:
                return json.dumps({})
            return json.dumps(results, ensure_ascii=False)

        else:
            return json.dumps({})
    except Exception as e:
        return json.dumps({})


if __name__ == '__main__':
    args = parse_args()
    print("result", func(args))
