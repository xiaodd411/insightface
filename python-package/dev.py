import cv2
import numpy as np
import insightface
import insightface.app as insapp
import insightface.data as insdata
from sklearn import preprocessing
import math

result = []


# 进行图片比对
def feature_compare(feature1, feature2):
    # 计算出欧氏距离传给dist
    diff = np.subtract(feature1, feature2)  # 使用矩阵相减 feature1 - feature2
    dist = np.sum(np.square(diff), 1)  # np.square()矩阵进行平方运算，全部变成正数，再将低于1的值相加，也就是dist值越低越有可能是同一个人
    return dist


# 提取两张（仅有一个人脸）的图片的人脸特征值存放到数组中
def get_face_feat(img):
    faces = app.get(img,max_num=1)
    feature = ()
    for face in faces:
        # print('face:',face)
        print('face.embedding:', face.embedding)
        print('face.normed_embedding:', face.normed_embedding)
        feature = np.array(face.embedding).reshape((1, -1))
        feature = preprocessing.normalize(feature)
        box = face.bbox.astype(np.int64)
        result.append(feature)


app = insapp.FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))
img = insdata.get_image('0')  # 不用带后缀，图片放到./insightface/python-package/insightface/data/images
img2 = insdata.get_image('5')

get_face_feat(img)
get_face_feat(img2)
res = []
dist = feature_compare(result[0], result[1])

print('dist:', dist)  # 值越小是同一个人的概率越大，我设置的阈值为 1
# if int(dist) <= 1:
#     print("是同一个人！")
# if int(dist) > 1:
#     print("不是同一个人！")
