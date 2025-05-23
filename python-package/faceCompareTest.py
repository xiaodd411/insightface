import os
import shutil

import cv2
import pickle
import logging
import numpy as np
from scipy.spatial.distance import euclidean, cityblock, minkowski, correlation, jaccard, hamming, chebyshev
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis


# 创建人脸分析对象
# app =  FaceAnalysis(allowed_modules=['detection', 'recognition'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], download=False)
app = FaceAnalysis(allowed_modules=['detection', 'recognition'], providers=[ 'CPUExecutionProvider'], download=False)
app.prepare(ctx_id=-1) # ctx_id=0 使用GPU，-1 使用CPU

# 设置日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 数据库文件路径
db_file = 'face_db_compare_test.pkl'

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
                user_id = f"{file_name}"
                db[user_id] = feature
                logging.info(f"已保存人脸特征: {user_id}")

# 处理待识别的文件夹，输出识别结果
def recognize_faces_in_folder(input_folder, db):
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)

        # 判断文件是否为图片文件
        if os.path.splitext(file_name.lower())[1] not in {".jpg", ".jpeg", ".png"}:
            continue

        face_features = extract_faces(file_path)
        if not face_features:
            logging.warning(f"图片: {file_name}没有人脸信息")
            continue

        for feature, idx in face_features:
            # 进行人脸比对
            max_similarity = 0
            best_match_user = None
            for user_id, registered_feature in db.items():
                similarity = cosine_similarity([feature], [registered_feature])[0][0]
                fixed_length_path = user_id.ljust(10)
                # logging.info(f"与 {fixed_length_path} 的相似度: {similarity:.2f}")

                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match_user = user_id

            # 判断是否为同一人
            similarity_threshold = 0.5
            similarity_threshold = -1.0
            if max_similarity >= similarity_threshold:
                fixed_length_path = best_match_user
                if best_match_user is not None:
                    fixed_length_path = best_match_user.ljust(10)
                logging.info(f" {file_name}的识别结果：{fixed_length_path}，相似度：{max_similarity:.4f}")
                if max_similarity >= 0.32:
                    # 将该图片移动到识别结果文件夹，文件名以当前文件名+识别结果+相似度 命名 识别结果文件夹在当前文件夹的名字后加上"_result"
                    result_folder = input_folder + "_recognition_result"
                    if not os.path.exists(result_folder):
                        os.makedirs(result_folder)
                    shutil.copy(file_path, os.path.join(result_folder, f"{file_name}_{best_match_user}_{max_similarity:.4f}.jpg"))
            else:
                logging.info("无法识别此人，可能是陌生人。")

if __name__ == "__main__":
    # 设置数据库路径和图片文件夹路径
    database_folder = 'test\\证件照'  # 这是存储数据库的文件夹
    to_recognize_folder = 'test\\11'  # 待识别的图片文件夹\\xq\\Desktop\\project\\insightface\\python-package\\test\\证件照'  # 这是存储数据库的文件夹
    to_recognize_folder = 'test\\11_correction'


    # 加载或初始化数据库
    face_db = load_face_db()

    # 处理数据库文件夹，更新数据库
    # process_database_folder(face_db, database_folder)
    save_face_db(face_db)

    # 处理待识别文件夹并输出识别结果
    recognize_faces_in_folder(to_recognize_folder, face_db)
    input_folder = input("按回车键结束: ").strip()
