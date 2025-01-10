import time

import cv2
import os
import logging
from insightface.app import FaceAnalysis
from contextlib import redirect_stdout, redirect_stderr

# 设置 ONNX Runtime 的日志级别
os.environ["ORT_LOG_LEVEL"] = "FATAL"

def initialize_face_analysis():
    """初始化并返回配置的 FaceAnalysis 实例。"""
    app = FaceAnalysis(allowed_modules=['detection'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], download=False)
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

def process_images(input_folder):
    # 初始化日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 初始化 FaceAnalysis 应用
    app = initialize_face_analysis()

    # 定义有效的图片扩展名集合
    valid_extensions = {".jpg", ".jpeg", ".png"}

    # 无人脸图片列表
    no_face_images = []

    # 遍历输入路径下的所有图片文件
    startTime = time.time()
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)

        # 检查是否为图片文件
        if os.path.splitext(file_name.lower())[1] not in valid_extensions:
            continue

        # 读取图片
        img = cv2.imread(file_path)
        if img is None:
            logging.error(f"无法读取图片: {file_name}")
            continue

        # 获取人脸信息
        faces = app.get(img)
        if not faces:
            logging.warning(f"未检测到人脸: {file_name}")
            no_face_images.append(file_name)
            continue
    endTime = time.time()
    # 记录花费时间
    logging.info(f"总花费时间: {endTime - startTime:.2f} 秒")
    # 打印无人脸的图片名称
    logging.info("以下图片未检测到人脸:")
    # for no_face_image in no_face_images:
    logging.info("以下图片未检测到人脸: " + ", ".join(no_face_images))

    # 输出一下目前结果
    logging.info(f"总图片数: {len(os.listdir(input_folder))}")
    logging.info(f"识别人脸的图片数: {len(os.listdir(input_folder)) - len(no_face_images)}")
    logging.info(f"识别率: {(len(os.listdir(input_folder)) - len(no_face_images)) / len(os.listdir(input_folder)):.2f}")
    logging.info(f"漏检率: {len(no_face_images) / len(os.listdir(input_folder)):.2f}")

    # 显示无人脸的图片并手动标记
    detected_faces = 0
    total_no_face_images = len(no_face_images)
    # for idx, no_face_image in enumerate(no_face_images):
    #     img = cv2.imread(os.path.join(input_folder, no_face_image))
    #     cv2.imshow(no_face_image, img)
    #     print(f"剩余图片数量: {total_no_face_images - idx - 1}")
    #     key = cv2.waitKey(0)
    #     if key == 13:  # Enter key
    #         detected_faces += 1
    #     elif key == 27:  # Esc key
    #         print("检测结束，剩余图片默认无人脸处理")
    #         break
    #     elif key == ord('f'):  # 'f' key for fast forward
    #         print("快进模式，剩余图片默认无人脸处理")
    #         break
    #     cv2.destroyAllWindows()

    # 计算识别率和漏检率
    total_images = len(os.listdir(input_folder))
    no_face_count = len(no_face_images)
    face_count = total_images - no_face_count + detected_faces
    recognition_rate = face_count/ total_images
    miss_rate = no_face_count / total_images

    logging.info(f"总图片数: {total_images}")
    logging.info(f"识别人脸的图片数: {face_count}")
    logging.info(f"识别率: {recognition_rate:.2f}")
    logging.info(f"漏检率: {miss_rate:.2f}")

    # 移动没有人脸的图片到同级另一个文件夹
    # no_face_folder = os.path.join("C:\\Users\\xq\\Desktop\\project\\insightface\\python-package\\test\\", "11-noFace")
    # os.makedirs(no_face_folder, exist_ok=True)
    # for no_face_image in no_face_images:
    #     os.rename(os.path.join(input_folder, no_face_image), os.path.join(no_face_folder, no_face_image))

if __name__ == "__main__":

    input_folder = input("请输入要处理的图片文件夹路径: ").strip()
    if not os.path.isdir(input_folder):
        logging.error("输入的路径无效，请提供有效的文件夹路径。")
    else:
        process_images(input_folder)
