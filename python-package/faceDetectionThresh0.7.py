import cv2
import os
import logging
from insightface.app import FaceAnalysis

# 设置 ONNX Runtime 的日志级别
os.environ["ORT_LOG_LEVEL"] = "FATAL"

def initialize_face_analysis():
    """初始化并返回配置的 FaceAnalysis 实例。"""
    app = FaceAnalysis(allowed_modules=['detection'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], download=False)
    app.prepare(ctx_id=0, det_size=(640, 640),det_thresh=0.7)
    return app

def process_images(input_folder):
    # 初始化日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 初始化 FaceAnalysis 应用
    app = initialize_face_analysis()

    # 定义有效的图片扩展名集合
    valid_extensions = {".jpg", ".jpeg", ".png"}

    no_face_images = []
    total_images = 0
    face_images = 0

    output_folder = f"{input_folder}_det_thresh_{app.det_thresh}"
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入路径下的所有图片文件
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)

        # 检查是否为图片文件
        if os.path.splitext(file_name.lower())[1] not in valid_extensions:
            continue

        total_images += 1

        # 读取图片
        img = cv2.imread(file_path)
        if img is None:
            logging.error(f"无法读取图片: {file_name}")
            continue

        # 获取人脸信息
        faces = app.get(img)
        if not faces:
            logging.warning(f"未检测到人脸: {file_name}")
            no_face_images.append(file_path)
        else:
            # 输出人脸检测的分数
            for face in faces:
                logging.info(f"{file_name}人脸检测分数: {face.det_score:.2f}")
            face_images += 1
            # 将人脸信息的文件复制一份到另一个文件夹
            output_path = os.path.join(output_folder, file_name)
            cv2.imwrite(output_path, img)

    # # 打印出无人脸的图片名称并依次显示
    # for img_path in no_face_images:
    #     img = cv2.imread(img_path)
    #     cv2.imshow("No Face Detected", img)
    #     key = cv2.waitKey(0)
    #     if key == 13:  # Enter key
    #         face_images += 1
    #     cv2.destroyAllWindows()

    # 计算识别率和漏检率
    recognition_rate = (face_images / total_images) * 100 if total_images > 0 else 0
    miss_rate = ((total_images - face_images) / total_images) * 100 if total_images > 0 else 0

    logging.info(f"总图片数: {total_images}")
    logging.info(f"识别到人脸的图片数: {face_images}")
    logging.info(f"识别率: {recognition_rate:.2f}%")
    logging.info(f"漏检率: {miss_rate:.2f}%")

if __name__ == "__main__":
    input_folder = input("请输入要处理的图片文件夹路径: ").strip()
    if not os.path.isdir(input_folder):
        logging.error("输入的路径无效，请提供有效的文件夹路径。")
    else:
        process_images(input_folder)
