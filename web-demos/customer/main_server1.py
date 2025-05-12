import os
import json
import asyncio
import logging
import cv2
import numpy as np
import base64
from fastapi import FastAPI, UploadFile, File
import uvicorn
from insightface.app import FaceAnalysis
from numpy.linalg import norm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)

app = FastAPI()
http_port = int(os.getenv("HTTP_PORT", "8066"))
detection_thresh = float(os.getenv("DETECTION_THRESH", "0.65"))
default_scale = float(os.getenv("CROP_SCALE", "1.8"))  # 默认裁剪比例

# 限制线程池，防止过多线程创建
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))  # 线程池大小，默认为 4，可调整
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# 初始化人脸分析器
face_analysis = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                             allowed_modules=['detection', 'recognition'])
face_analysis.prepare(ctx_id=0, det_thresh=detection_thresh, det_size=(640, 640))

@app.get("/")
async def root():
    return {"message": "Face Detection API is running."}

def expand_bbox(bbox, scale, img_width, img_height):
    """扩展人脸框并确保不会超出图像边界"""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    new_width = int(width * scale)
    new_height = int(height * scale)

    new_x1 = max(x1 - (new_width - width) // 2, 0)
    new_y1 = max(y1 - (new_height - height) // 2, 0)
    new_x2 = min(new_x1 + new_width, img_width)
    new_y2 = min(new_y1 + new_height, img_height)

    return [new_x1, new_y1, new_x2, new_y2]

async def process_image(file: UploadFile, scale: float):
    """异步处理单个文件，包括人脸检测和裁剪"""
    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"filename": file.filename, "result": [], "message": "Invalid or corrupted image file."}

    # 使用线程池执行人脸检测
    loop = asyncio.get_running_loop()
    faces = await loop.run_in_executor(executor, face_analysis.get, img)

    file_results = []
    img_height, img_width = img.shape[:2]

    for face in faces:
        bbox = list(map(int, face.bbox))
        expanded_bbox = expand_bbox(bbox, scale, img_width, img_height)
        cropped_face = img[expanded_bbox[1]:expanded_bbox[3], expanded_bbox[0]:expanded_bbox[2]]
        _, buffer = cv2.imencode('.jpg', cropped_face)
        face_base64 = base64.b64encode(buffer).decode('utf-8')

        embedding = face.normed_embedding.astype(float)
        file_results.append({
            "filename": file.filename,
            "found": True,
            "score": float(face.det_score),
            "feature": (embedding / norm(embedding)).tolist(),
            "facial_area": {
                "x": expanded_bbox[0], "y": expanded_bbox[1], "w": expanded_bbox[2] - expanded_bbox[0], "h": expanded_bbox[3] - expanded_bbox[1]
            },
            "path": face_base64
        })

    return {"result": file_results}

@app.post("/detect_and_crop")
async def detect_and_crop_faces(files: list[UploadFile] = File(...), scale: float = default_scale):
    """异步人脸检测与裁剪，使用线程池"""
    tasks = [process_image(file, scale) for file in files]
    results = await asyncio.gather(*tasks)
    return {"results": results}

async def process_representation(file: UploadFile):
    """异步处理人脸特征提取"""
    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"filename": file.filename, "result": [], "message": "Invalid or corrupted image file."}

    # 使用线程池执行人脸识别
    loop = asyncio.get_running_loop()
    faces = await loop.run_in_executor(executor, face_analysis.get, img)

    file_results = []
    for face in faces:
        embedding = face.normed_embedding.astype(float)
        file_results.append({
            "filename": file.filename,
            "found": True,
            "score": float(face.det_score),
            "feature": (embedding / norm(embedding)).tolist(),
            "facial_area": {
                "x": int(face.bbox[0]), "y": int(face.bbox[1]), "w": int(face.bbox[2] - face.bbox[0]), "h": int(face.bbox[3] - face.bbox[1])
            }
        })

    return {"result": file_results}

@app.post("/detect")
async def represent_faces(files: list[UploadFile] = File(...)):
    """异步人脸特征提取，使用线程池"""
    tasks = [process_representation(file) for file in files]
    results = await asyncio.gather(*tasks)
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=http_port)
