import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import models, layers
import matplotlib.pyplot as plt

# MediaPipe 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 모델 로드 (예: TensorFlow 모델)
model = tf.keras.models.load_model('squat.h5')

# 카운트 및 상태 변수 초기화
squat_count = 0
current_state = "up"  # 초기 상태: 스쿼트 시작 전

def preprocess_landmarks(landmarks):
    # MediaPipe에서 추출된 랜드마크를 AI 모델 입력 형태로 변환
    keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
    return np.expand_dims(keypoints, axis=0)

# 비디오 파일 경로 설정
video_path = r'C:\Users\admin\Desktop\모블_텐서플로우\prj_Exercise-Posture\Videos\Squrt.mp4'

# OpenCV 비디오 캡처 설정
cap = cv2.VideoCapture(video_path)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR을 RGB로 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # MediaPipe로 포즈 감지
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            # 랜드마크를 그리기
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # 랜드마크 전처리
            input_data = preprocess_landmarks(results.pose_landmarks.landmark)

            # 모델 예측
            prediction = model.predict(input_data)
            predicted_class = np.argmax(prediction)  # 0: Down, 1: 오류

            # 상태 전환 및 카운트 관리
            if predicted_class == 0 and current_state == "up":
                squat_count += 1
                current_state = "down"
                print("스쿼트 카운트:", squat_count)
            elif predicted_class == 1:
                print("자세 오류: 무릎을 더 구부리세요.")
            elif predicted_class == 0:
                current_state = "down"
            else:
                current_state = "up"

        # 화면에 카운트 표시
        cv2.putText(image, f'Squat Count: {squat_count}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Squat Tracker', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
