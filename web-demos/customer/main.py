import cv2
import os
import json
import onnxruntime
from insightface.app import FaceAnalysis
from typing import List
from numpy.linalg import norm

# 设置 ONNX Runtime 的日志级别
os.environ["ORT_LOG_LEVEL"] = "FATAL"
onnxruntime.set_default_logger_severity(3)

def initialize_face_analysis():
    """初始化并返回配置的 FaceAnalysis 实例。"""
    app = FaceAnalysis(allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'], download=False)
    app.prepare(ctx_id=0, det_thresh=0.1, det_size=(640, 640))
    return app

def expand_bbox(bbox, scale=1.2, img_width=None, img_height=None):
    """扩展人脸框并确保不会超出图像边界"""
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

def save_cropped_face(img, face, image_name, output_dir, face_index):
    """裁剪人脸并保存到指定目录"""
    bbox = face.bbox.astype(int)  # 人脸框坐标 [x1, y1, x2, y2]

    # 获取图像尺寸
    img_height, img_width = img.shape[:2]

    # 扩展边界，增加 20% 的边界
    expanded_bbox = expand_bbox(bbox, scale=1.8, img_width=img_width, img_height=img_height)

    # 裁剪人脸部分
    cropped_face = img[expanded_bbox[1]:expanded_bbox[3], expanded_bbox[0]:expanded_bbox[2]]  # 裁剪区域

    # 创建文件夹以存储裁剪的图片
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 使用人脸顺序命名裁剪图片
    # cropped_image_path = os.path.join(output_dir, f"{image_name}_face_{face_index + 1}.jpg")
    cropped_image_path = os.path.join(output_dir, f"{face_index + 1}.jpg")
    cv2.imwrite(cropped_image_path, cropped_face)  # 保存裁剪图片
    return cropped_image_path

def process_image(image_path, app, process_faces=False):
    result = []
    # 获取图片文件名称(不包含后缀)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {image_name: {"found": False, "score": None, "feature": None, "message": "无法读取图片"}}

        # 获取人脸信息
        faces = app.get(img)
        if not faces:
            return {image_name: {"found": False, "score": None, "feature": None, "message": "未检测到人脸"}}

        # 创建一个存储裁剪图像的文件夹
        output_dir = os.path.join(os.path.dirname(image_path), image_name)
        cropped_paths = []

        for i, face in enumerate(faces):
            bbox = face.bbox.astype(int)  # 人脸框坐标 [x1, y1, x2, y2]
            score = face.det_score  # 人脸得分
            feature = face.normed_embedding  # 归一化的人脸特征向量
            result_entry = {
                "found": True,
                "score": float(score),
                "feature": (feature / norm(feature)).tolist(),
                "originName": image_name,
            }
            # 如果需要处理裁剪功能
            if process_faces:
                cropped_image_path = save_cropped_face(img, face, i, output_dir, i)
                result_entry["path"] = cropped_image_path
                cropped_paths.append(cropped_image_path)

            result.append(result_entry)

        return {image_name: result}

    except Exception as e:
        result.append({"found": False, "score": None, "feature": None, "message": "处理图片时出错:"+str(e)})
        return {image_name: result}

def process_directory(directory_path, app, process_faces=False):
    """处理文件夹中的所有图片文件，返回每张图片的检测结果"""
    image_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    results = list(map(lambda f: process_image(f, app, process_faces), image_files))
    return results

def func(action, input_path):
    """根据输入的 action 和 input_path 执行相应的操作"""
    if action not in ['detection', 'detectionAndProcess']:
        print(json.dumps({}))
        return

    try:
        # 初始化 FaceAnalysis 应用
        app = initialize_face_analysis()

        # 判断是否需要裁剪人脸
        process_faces = action == 'detectionAndProcess'

        if os.path.isdir(input_path):
            results = process_directory(input_path, app, process_faces)
            result_json = {k: v for d in results for k, v in d.items()}  # 合并所有结果
            print("result",json.dumps(result_json, ensure_ascii=False))

        elif os.path.isfile(input_path):
            result = process_image(input_path, app, process_faces)
            print("result",json.dumps(result, ensure_ascii=False))

        else:
            # 输入路径无效，输出空的JSON map
            print("result",json.dumps({}))
            return
    except Exception as e:
        print(json.dumps({}))
        return -1

if __name__ == "__main__":
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, help="Action to perform, 'detection' for face detection or 'detectionAndProcess' for face detection and processing")
    parser.add_argument('input_path', type=str, help="Image file or directory path to process")
    args = parser.parse_args()

    # 调用处理函数
    func(args.action, args.input_path)
