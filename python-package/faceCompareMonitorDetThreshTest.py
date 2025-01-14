import os
import cv2
import pickle
import logging
import numpy as np
from scipy.spatial.distance import euclidean, cityblock, minkowski, correlation, jaccard, hamming, chebyshev
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
import os
import shutil
import logging
from collections import defaultdict


# 创建人脸分析对象
app =  FaceAnalysis(allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'], download=False)
# app = FaceAnalysis()
app.prepare(ctx_id=-1,det_thresh=0.4)  # ctx_id=0 使用GPU，-1 使用CPU

# 设置日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 数据库文件路径
db_file = 'faceCompareMonitorDetThreshTest.pkl'

# 加载或创建人脸数据库
def load_face_db():
    try:
        with open(db_file, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        logging.info("数据库文件不存在，将创建新数据库")
        return {}

# 保存数据库
def save_face_db(db):
    with open(db_file, 'wb') as f:
        pickle.dump(db, f)

# 提取图片中的人脸特征
def extract_faces(img_path):
    # img = cv2.imread(img_path)
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR) # 解决imread不能读取中文路径的问题
    if img is None:
        logging.error(f"无法读取图片: {img_path}")
        return None

    faces = app.get(img)
    if not faces:
        logging.warning(f"未检测到人脸: {img_path}")
        return None

    face_features = []
    for i, face in enumerate(faces):
        # feature = face.embedding  # 提取人脸特征
        feature = face.normed_embedding  # 提取人脸特征的归一化向量
        # 输出人脸特征的归一化向量
        # print(f"人脸特征的归一化向量: {feature}")
        # feature = face.embedding_norm  # 提取人脸特征
        face_features.append((feature, i))

    return face_features

# 处理数据库中的文件，更新数据库
def process_database_folder(db, input_folder):
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)

        # 判断文件是否为图片文件
        if os.path.splitext(file_name.lower())[1] not in {".jpg", ".jpeg", ".png"}:
            continue

        logging.info(f"正在处理图片: {file_name}")
        face_features = extract_faces(file_path)
        if face_features:
            for feature, idx in face_features:
                db[file_path] = feature

# 处理待识别的文件夹，输出识别结果
def recognize_faces_in_folder(db, output_folder, threshold):
    """
    将数据库中的人脸进行一一比对，归类到不同的文件夹中，并统计不同人脸的数量。

    参数:
        db (dict): 键为文件路径，值为人脸归一化向量。
        output_folder (str): 输出文件夹路径。
        threshold (float): 人脸相似度阈值，默认值为 0.6。
    """
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 记录已经处理过的文件路径
    processed_files = set()

    # 用于统计不同人脸的数量
    unique_face_count = 0

    # 遍历数据库中的每个人脸特征
    for file_path, feature in db.items():
        # 如果当前文件已经处理过，跳过
        if file_path in processed_files:
            continue

        # 创建一个新的文件夹用于存储当前人脸的图片
        unique_face_count += 1
        face_folder = os.path.join(output_folder, f"face_{unique_face_count}")
        if not os.path.exists(face_folder):
            os.makedirs(face_folder)

        # 记录当前文件路径已经处理过
        processed_files.add(file_path)

        # 将当前人脸的图片复制到文件夹中
        image_name = os.path.basename(file_path)
        new_image_path = os.path.join(face_folder, image_name)
        shutil.copy(file_path, new_image_path)
        logging.info(f"图片已复制: {new_image_path}")

        # 将特征向量转换为 2D 数组
        feature = np.array(feature).reshape(1, -1)  # 将 1D 向量转换为 2D 数组

        # 与数据库中的其他人脸特征进行比对，找到相似的人脸
        for other_file_path, other_feature in db.items():
            # 如果其他人脸已经处理过，跳过
            if other_file_path in processed_files:
                continue

            # 将其他人脸特征向量转换为 2D 数组
            other_feature = np.array(other_feature).reshape(1, -1)  # 将 1D 向量转换为 2D 数组

            # 计算特征向量的相似度
            distance = cosine_similarity(feature, other_feature)

            # 如果相似度超过阈值，认为是同一个人
            if distance > threshold:  # 注意：这里使用 >，因为余弦相似度越大越相似
                # 将相似的人脸归类到同一个文件夹中
                processed_files.add(other_file_path)
                other_image_name = os.path.basename(other_file_path)
                new_other_image_path = os.path.join(face_folder, other_image_name)
                shutil.copy(other_file_path, new_other_image_path)
                logging.info(f"图片已复制: {new_other_image_path}")

    # 输出不同人脸的数量
    logging.info(f"数据库中不同人脸的数量: {unique_face_count}")
    print(f"数据库中不同人脸的数量: {unique_face_count}")

if __name__ == "__main__":
    # 设置数据库路径和图片文件夹路径
    database_folder = 'F:\\project\\insightface\\python-package\\test\\11_det_thresh_0.7'  # 这是存储数据库的文件夹

    # 加载或初始化数据库
    face_db = load_face_db()

    # 处理数据库文件夹，更新数据库
    process_database_folder(face_db, database_folder)
    save_face_db(face_db)

    threshold=0.60
    # 处理待识别文件夹并输出识别结果
    recognize_faces_in_folder(face_db,f'F:\\project\\insightface\\python-package\\test\\11_det_thresh_0.7_recognize_thresh_{threshold:.2f}_res',threshold)
    # input_folder = input("按回车键结束: ").strip()
