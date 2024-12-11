import cv2

from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity


# 定义数据库文件路径
db_file = 'face_db.pkl'

# 创建人脸分析对象
app = FaceAnalysis()
# 加载预训练模型
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0表示使用GPU，-1表示使用CPU

# 加载图片进行分析
img_path = 'E:\\Desktop\\tetst\\0000000000000000241107172225647000000000000057514.jpg'

img = cv2.imread(img_path)
#
# img = img[:, :, ::-1]

# img = ins_get_image('test3')
# 提取人脸信息
faces = app.get(img)

# 输出提取到的人脸信息
for face in faces:
    print(f"人脸位置: {face.bbox}")  # 人脸框
    print(f"特征向量: {face.embedding}")  # 人脸特征向量

# 假设我们只处理检测到的第一个人脸
if len(faces) > 0:
    face = faces[0]
    feature = face.embedding  # 这是一个128维的人脸特征向量
    print("人脸特征：", feature)

# 假设 db_features 是数据库中存储的人脸特征
# 比较数据库中某个特征与提取到的特征
similarity = cosine_similarity([feature], [db_feature])
print(f"相似度：{similarity[0][0]}")

# 绘制检测到的人脸框
for face in faces:
    bbox = face.bbox  # 人脸框
    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

    # 显示相似度
    cv2.putText(img, f"Similarity: {similarity[0][0]:.2f}",
                (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# 显示图片
cv2.imshow('Face Recognition', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
