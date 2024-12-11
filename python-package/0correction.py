import cv2
import os
import logging
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

def initialize_face_analysis():
    """初始化并返回配置的 FaceAnalysis 实例。"""
    app = FaceAnalysis(allowed_modules=['detection'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], download=False)
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

def process_images(input_folder, output_folder=None):
    # 初始化日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 初始化 FaceAnalysis 应用
    app = initialize_face_analysis()

    # 如果未提供输出文件夹名称，设置默认值
    if output_folder is None:
        output_folder = os.path.join(input_folder, "processed_faces")

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 定义有效的图片扩展名集合
    valid_extensions = {".jpg", ".jpeg", ".png"}

    # 遍历输入路径下的所有图片文件
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
            continue

        # 遍历检测到的人脸并剪切保存
        for i, face in enumerate(faces):
            # 获取人脸位置
            bbox = face.bbox.astype(int)  # 人脸框坐标 [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox

            # 适当增加边界（例如扩大 20%）
            width = x2 - x1
            height = y2 - y1
            padding_x = int(0.1 * width)  # 增加 20% 宽度
            padding_y = int(0.1 * height)  # 增加 20% 高度

            # 计算新的边界，确保不会超出图片范围
            x1 = max(0, x1 - padding_x)
            y1 = max(0, y1 - padding_y)
            x2 = min(img.shape[1], x2 + padding_x)
            y2 = min(img.shape[0], y2 + padding_y)

            # 裁剪人脸区域
            face_img = img[y1:y2, x1:x2]

            # 生成保存文件名
            base_name, ext = os.path.splitext(file_name)
            if len(faces) > 1:
                output_file_name = f"{base_name}_correction_{i + 1}{ext}"
            else:
                output_file_name = f"{base_name}_correction{ext}"

            output_file_path = os.path.join(output_folder, output_file_name)

            # 保存人脸图片
            cv2.imwrite(output_file_path, face_img)
            logging.info(f"人脸已保存: {output_file_path}")

    logging.info(f"处理完成，所有人脸图片保存在: {output_folder}")

if __name__ == "__main__":
    input_folder = input("请输入要处理的图片文件夹路径: ").strip()
    if not os.path.isdir(input_folder):
        logging.error("输入的路径无效，请提供有效的文件夹路径。")
    else:
        output_folder = input("请输入输出文件夹路径（或留空以使用默认值）: ").strip()
        process_images(input_folder, output_folder if output_folder else None)
