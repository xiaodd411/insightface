import os
import sys
import json
import asyncio
import logging
import cv2
import numpy as np
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
import uvicorn
from insightface.app import FaceAnalysis
from numpy.linalg import norm
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)

app = FastAPI()
http_port = int(os.getenv("HTTP_PORT", "8066"))
detection_thresh = float(os.getenv("DETECTION_THRESH", "0.65"))
default_scale = float(os.getenv("CROP_SCALE", "1.8"))  # 默认裁剪比例

# 初始化人脸分析器
face_analysis = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                             allowed_modules=['detection', 'recognition'])
face_analysis.prepare(ctx_id=0, det_thresh=detection_thresh, det_size=(640, 640))

# 创建线程池执行器
executor = ThreadPoolExecutor(max_workers=12)

@app.on_event("shutdown")
def shutdown_event():
    executor.shutdown(wait=False)

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

async def process_single_file(file, scale, crop=False):
    try:
        image_bytes = await file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return {"filename": file.filename, "result": [], "message": "Invalid image"}

        # 将同步的face_analysis.get放入线程池执行
        loop = asyncio.get_event_loop()
        faces = await loop.run_in_executor(executor, face_analysis.get, img)

        file_results = []
        img_height, img_width = img.shape[:2]

        for face in faces:
            if crop:
                bbox = list(map(int, face.bbox))
                expanded_bbox = expand_bbox(bbox, scale, img_width, img_height)
                cropped_face = img[expanded_bbox[1]:expanded_bbox[3], expanded_bbox[0]:expanded_bbox[2]]
                _, buffer = cv2.imencode('.jpg', cropped_face)
                face_base64 = base64.b64encode(buffer).decode('utf-8')
            else:
                face_base64 = None

            embedding = face.normed_embedding.astype(float)
            result = {
                "filename": file.filename,
                "found": True,
                "score": float(face.det_score),
                "feature": (embedding / norm(embedding)).tolist(),
                "facial_area": {
                    "x": expanded_bbox[0] if crop else int(face.bbox[0]),
                    "y": expanded_bbox[1] if crop else int(face.bbox[1]),
                    "w": (expanded_bbox[2] - expanded_bbox[0]) if crop else int(face.bbox[2] - face.bbox[0]),
                    "h": (expanded_bbox[3] - expanded_bbox[1]) if crop else int(face.bbox[3] - face.bbox[1])
                }
            }
            if face_base64:
                result["path"] = face_base64
            file_results.append(result)

        return {"result": file_results}
    except Exception as e:
        logging.error(f"Error processing {file.filename}: {str(e)}")
        return {"filename": file.filename, "result": [], "message": str(e)}

@app.post("/detect_and_crop")
async def detect_and_crop_faces(files: list[UploadFile] = File(...), scale: float = default_scale):
    tasks = [process_single_file(file, scale, crop=True) for file in files]
    results = await asyncio.gather(*tasks)
    return {"results": results}

@app.post("/detect")
async def detect_faces(files: list[UploadFile] = File(...)):
    tasks = [process_single_file(file, scale=1.0, crop=False) for file in files]
    results = await asyncio.gather(*tasks)
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=http_port)
