from sklearn.metrics.pairwise import cosine_similarity
import pickle
import cv2
import insightface
from insightface.app import FaceAnalysis

# 创建人脸分析对象
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 使用GPU，-1 使用CPU

# 定义数据库文件路径
db_file = 'face_db.pkl'

# 读取待识别的图片
# img_path = 'E:\\Desktop\\tetst\\0000000000000000241107172225647000000000000057516.jpg'
# img_path = 'E:\\Desktop\\FaceRecognition\\lgh\\1.png'
img_path = 'E:\\Desktop\\FaceRecognition\\lgh\\zjz.jpg'
img = cv2.imread(img_path)

# 提取待识别图像的人脸特征
faces = app.get(img)

if len(faces) > 0:
    # 假设我们只处理检测到的第一个人脸
    feature_to_compare = faces[0].embedding  # 提取待比对的人脸特征

    # 加载注册的数据库
    try:
        with open(db_file, 'rb') as f:
            db = pickle.load(f)
    except FileNotFoundError:
        print("数据库文件不存在！")
        exit()

    # 在数据库中进行比对
    max_similarity = 0
    best_match_user = None
    for user_id, registered_feature in db.items():
        similarity = cosine_similarity([feature_to_compare], [registered_feature])[0][0]
        print(f"与 {user_id} 的相似度: {similarity:.2f}")

        # 找到相似度最高的用户
        if similarity > max_similarity:
            max_similarity = similarity
            best_match_user = user_id

    # 设置一个阈值判断是否为同一人（可以根据实际情况调整）
    similarity_threshold = 0.7

    if max_similarity >= similarity_threshold:
        print(f"识别结果：{best_match_user}，相似度：{max_similarity:.2f}")
    else:
        print("无法识别此人，可能是陌生人。")
else:
    print("未检测到人脸，无法识别。")
