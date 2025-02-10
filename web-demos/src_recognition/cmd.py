#!/usr/bin/env python

import os
import os.path as osp
import argparse
import cv2
import numpy as np
import onnxruntime
from scrfd import SCRFD
from arcface_onnx import ArcFaceONNX
import json

onnxruntime.set_default_logger_severity(3)

assets_dir = osp.expanduser('~/.insightface/models/buffalo_l')

detector = SCRFD(os.path.join(assets_dir, 'det_10g.onnx'))
detector.prepare(0)
model_path = os.path.join(assets_dir, 'w600k_r50.onnx')
rec = ArcFaceONNX(model_path)
rec.prepare(0)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    detect_parser = subparsers.add_parser('detect')
    detect_parser.add_argument('img', type=str)

    crop_parser = subparsers.add_parser('crop')
    crop_parser.add_argument('img', type=str)
    crop_parser.add_argument('output_dir', type=str)

    compare_parser = subparsers.add_parser('compare')
    compare_parser.add_argument('img1', type=str)
    compare_parser.add_argument('img2', type=str)

    return parser.parse_args()

def detect_faces(image_path):
    image = cv2.imread(image_path)
    bboxes, kpss = detector.autodetect(image)
    results = []

    for i, (bbox, kps) in enumerate(zip(bboxes, kpss)):
        score = bbox[4]
        feat = rec.get(image, kps)
        results.append({
            'index': i,
            'score': float(score),
            'feature': feat.tolist()
        })

    return results

def crop_faces(image_path, output_dir):
    image = cv2.imread(image_path)
    bboxes, kpss = detector.autodetect(image)
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for i, (bbox, kps) in enumerate(zip(bboxes, kpss)):
        x1, y1, x2, y2 = map(int, bbox[:4])
        face = image[y1:y2, x1:x2]
        output_path = osp.join(output_dir, f'{i}.jpg')
        cv2.imwrite(output_path, face)
        results.append({
            'index': i,
            'path': output_path
        })

    return results

def compare_faces(image1_path, image2_path):
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    bboxes1, kpss1 = detector.autodetect(image1)
    bboxes2, kpss2 = detector.autodetect(image2)

    if bboxes1.shape[0] == 0 or bboxes2.shape[0] == 0:
        return {'error': 'Face not found in one or both images'}

    results = []

    for (bbox1, kps1), (bbox2, kps2) in zip(zip(bboxes1, kpss1), zip(bboxes2, kpss2)):
        feat1 = rec.get(image1, kps1)
        feat2 = rec.get(image2, kps2)
        sim = rec.compute_sim(feat1, feat2)

        results.append({
            'similarity': float(sim),
            'image1': {
                'score': float(bbox1[4]),
                'feature': feat1.tolist()
            },
            'image2': {
                'score': float(bbox2[4]),
                'feature': feat2.tolist()
            }
        })

    return results

def main():
    args = parse_args()
    args.command = 'detect'
    args.img = "F:\project\insightface\python-package\data\discern_correction\\0_correction.jpg"
    if args.command == 'detect':
        results = detect_faces(args.img)
        print('result:', json.dumps(results))
    elif args.command == 'crop':
        results = crop_faces(args.img, args.output_dir)
        print('result:', json.dumps(results))
    elif args.command == 'compare':
        results = compare_faces(args.img1, args.img2)
        print('result:', json.dumps(results))

if __name__ == '__main__':
    main()
