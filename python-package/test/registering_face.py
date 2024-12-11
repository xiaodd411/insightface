import cv2
import numpy as np
import pickle
import insightface
from insightface.app import FaceAnalysis

# 创建人脸分析对象
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 使用GPU，-1 使用CPU

# 定义保存人脸特征的文件
db_file = 'face_db.pkl'

# 读取图片，假设图片中有待注册的用户
img_path = 'E:\Desktop\FaceRecognition\lgh\zjz.jpg'
img = cv2.imread(img_path)

# 提取人脸信息
faces = app.get(img)

if len(faces) > 0:
    # 假设我们只处理检测到的第一个人脸
    face = faces[0]
    feature = face.embedding  # 128维的人脸特征

    # 获取用户ID（或者可以是用户姓名等信息）
    user_id = '刘广华'

    # 加载数据库，或者初始化空数据库
    try:
        with open(db_file, 'rb') as f:
            db = pickle.load(f)
    except FileNotFoundError:
        db = {}

    # 将人脸特征与用户信息存储到数据库
    db[user_id] = feature

    # 保存数据库
    with open(db_file, 'wb') as f:
        pickle.dump(db, f)

    print(f"用户 {user_id} 的人脸信息已注册。")
