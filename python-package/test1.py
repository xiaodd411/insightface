import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

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
