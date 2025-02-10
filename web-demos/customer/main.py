import cv2
import os
import json
import onnxruntime
from insightface.app import FaceAnalysis
from typing import List

# 设置 ONNX Runtime 的日志级别
os.environ["ORT_LOG_LEVEL"] = "FATAL"
onnxruntime.set_default_logger_severity(3)

def initialize_face_analysis():
    """初始化并返回配置的 FaceAnalysis 实例。"""
    app = FaceAnalysis(allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'], download=False)
    app.prepare(ctx_id=0, det_thresh=0.1, det_size=(640, 640))
    return app

def process_image(image_path, app):
    """处理单张图片并返回检测结果，包括是否检测到人脸，得分，特征向量等"""
    img = cv2.imread(image_path)
    if img is None:
        return {image_path: {"face_found": False, "score": None, "feature": None, "message": "无法读取图片"}}

    # 获取人脸信息
    faces = app.get(img)
    if not faces:
        return {image_path: {"face_found": False, "score": None, "feature": None, "message": "未检测到人脸"}}

    # 遍历检测到的人脸并返回相关信息
    result = {}
    for i, face in enumerate(faces):
        bbox = face.bbox.astype(int)  # 人脸框坐标 [x1, y1, x2, y2]
        score = face.det_score  # 人脸得分
        feature = face.normed_embedding  # 归一化的人脸特征向量
        result = {
            "face_found": True,
            "score": score,
            "feature": feature.tolist(),  # 将 NumPy 数组转换为列表
            "message": "检测到人脸"
        }
    return {image_path: result}

def process_directory(directory_path, app):
    """处理文件夹中的所有图片文件，返回每张图片的检测结果"""
    image_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    results = list(map(lambda f: process_image(f, app), image_files))
    return results

def func(action, input_path):
    """根据输入的 action 和 input_path 执行相应的操作"""
    if action != 'detection':
        # 如果action不是detection, 直接输出空的JSON map
        print(json.dumps({}))
        return

    try:
        # 初始化 FaceAnalysis 应用
        app = initialize_face_analysis()

        # 如果是文件夹，处理文件夹中的所有图片
        if os.path.isdir(input_path):
            results = process_directory(input_path, app)
            result_json = {k: v for d in results for k, v in d.items()}  # 合并所有结果
            print(json.dumps(result_json, ensure_ascii=False, indent=4))

        # 如果是单个文件，处理该图片
        elif os.path.isfile(input_path):
            result = process_image(input_path, app)
            print(json.dumps(result, ensure_ascii=False, indent=4))

        else:
            # 输入路径无效，输出空的JSON map
            print(json.dumps({}))
            return
    except Exception as e:
        # 出现错误时，输出空的JSON map
        print(json.dumps({}))
        return -1

if __name__ == "__main__":
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, help="Action to perform, 'detection' for face detection")
    parser.add_argument('input_path', type=str, help="Image file or directory path to process")
    args = parser.parse_args()

    # 调用处理函数
    func(args.action, args.input_path)
