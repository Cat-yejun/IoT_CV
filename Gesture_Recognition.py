# gesture_recognition
import cv2
import numpy as np
from Configuration import gesture

def is_pointing_right(joint):
    # 손목(관절 0)과 검지 끝(관절 8)의 벡터 계산
    wrist = joint[0]
    index_finger_tip = joint[8]

    # 손목에서 손가락 끝까지의 벡터
    vector = index_finger_tip - wrist

    # 벡터의 x축 성분이 양수이면 오른쪽을 가리키고 있다고 판단
    if vector[0] > 0:  # x축이 양수이면 오른쪽으로 가리킴
        return True
    return False

def is_pointing_left(joint):
    # 손목(관절 0)과 검지 끝(관절 8)의 벡터 계산
    wrist = joint[0]
    index_finger_tip = joint[8]

    # 손목에서 손가락 끝까지의 벡터
    vector = index_finger_tip - wrist

    # 벡터의 x축 성분이 음수이면 왼쪽을 가리킴
    if vector[0] < 0:  # x축이 음수이면 왼쪽으로 가리킴
        return True
    return False

def is_pointing_up(joint):
    # 손목(관절 0)과 검지 끝(관절 8)의 벡터 계산
    wrist = joint[0]
    index_finger_tip = joint[8]

    # 손목에서 손가락 끝까지의 벡터
    vector = index_finger_tip - wrist

    # 벡터의 y축 성분이 음수이면 위쪽을 가리킴
    if vector[1] < 0:  # y축이 음수이면 위쪽으로 가리킴
        return True
    return False

def is_pointing_down(joint):
    # 손목(관절 0)과 검지 끝(관절 8)의 벡터 계산
    wrist = joint[0]
    index_finger_tip = joint[8]

    # 손목에서 손가락 끝까지의 벡터
    vector = index_finger_tip - wrist

    # 벡터의 y축 성분이 양수이면 아래쪽을 가리킴
    if vector[1] > 0:  # y축이 양수이면 아래쪽으로 가리킴
        return True
    return False




def load_gesture_model(file_path='gesture_train.csv'):
    file = np.genfromtxt(file_path, delimiter=',')
    angle = file[:, :-1].astype(np.float32)
    label = file[:, -1].astype(np.float32)
    knn = cv2.ml.KNearest_create()
    knn.train(angle, cv2.ml.ROW_SAMPLE, label)
    return knn

def recognize_direction(joint):
    if is_pointing_right(joint):
        return 'right'
    if is_pointing_left(joint):
        return 'left'
    if is_pointing_down(joint):
        return 'down'
    if is_pointing_up(joint):
        return 'up'

def recognize_gesture(knn, joint):
   

    # 나머지 기존 제스처 인식 로직
    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
    v = v2 - v1
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

    angle = np.arccos(np.einsum('nt,nt->n',
                                v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
    angle = np.degrees(angle)

    # Inference gesture
    data = np.array([angle], dtype=np.float32)
    ret, results, neighbours, dist = knn.findNearest(data, 3)
    idx = int(results[0][0])

    if idx in gesture.keys():
        return gesture[idx]
    return None
