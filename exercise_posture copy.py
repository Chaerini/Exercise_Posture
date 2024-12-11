import cv2
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# MediaPipe Pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# MediaPipe Pose Connection 설정
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 예시 데이터셋 (각도, 각도 범위, 자세 상태)
data = {
    'knee_angle': [100, 75, 120, 130, 90, 85],
    'hip_angle': [90, 85, 80, 85, 90, 95],
    'correct_posture': [1, 0, 1, 0, 1, 0]  # 1: 올바른 자세, 0: 잘못된 자세
}
df = pd.DataFrame(data)

# 특징과 라벨 분리
X = df.drop('correct_posture', axis=1)  # 특징
y = df['correct_posture']  # 라벨

# 훈련 데이터와 테스트 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델 생성
model = RandomForestClassifier()

# 모델 훈련
model.fit(X_train, y_train)

# 모델 예측
y_pred = model.predict(X_test)

# 모델 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"모델 정확도: {accuracy * 100:.2f}%")

# 비디오 파일 경로 설정
video_path = r'C:\Users\admin\Desktop\모블_텐서플로우\prj_Exercise-Posture\Videos\Squrt.mp4'

# OpenCV 비디오 캡처 설정
cap = cv2.VideoCapture(video_path)

# 각도 계산 함수 (예시로 구현)
def calculate_knee_angle(landmarks):
    landmarks = 90
    # 두 점을 이용하여 각도를 계산하는 로직을 구현
    # 예시로 무릎 각도를 추정하는 방식 (실제로는 정확한 계산 필요)
    # landmarks[25] -> 왼쪽 무릎, landmarks[26] -> 오른쪽 무릎
    return 100  # 예시 값, 실제 구현 시 각도 계산 필요

def calculate_hip_angle(landmarks):
    landmarks = 100
    # 엉덩이 각도 계산 로직
    # landmarks[23] -> 왼쪽 엉덩이, landmarks[24] -> 오른쪽 엉덩이
    return 90  # 예시 값, 실제 구현 시 각도 계산 필요


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 색상 변환: BGR -> RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Pose 모델 추적
    results = pose.process(rgb_frame)

    # 결과가 있는 경우
    if results.pose_landmarks:
        # 랜드마크와 관절 연결선 그리기
        mp_drawing.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,  # 관절을 선으로 연결
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            connection_drawing_spec=mp_drawing_styles.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        )
        
        # Pose 키포인트 추출 (예: 무릎 각도, 엉덩이 각도)
        landmarks = results.pose_landmarks.landmark
        knee_angle = calculate_knee_angle(landmarks)  # 무릎 각도 계산
        hip_angle = calculate_hip_angle(landmarks)    # 엉덩이 각도 계산
        
        # 머신러닝 모델을 사용하여 잘못된 자세 예측
        new_data = np.array([[knee_angle, hip_angle]])  # 예측할 새로운 데이터 (무릎 각도, 엉덩이 각도)
        prediction = model.predict(new_data)

        # 잘못된 자세에 대한 교정 피드백 제공
        if prediction == 1:
            print("올바른 자세입니다.")
        else:
            # 잘못된 자세일 경우, 피드백 제공
            if knee_angle < 90:
                print("무릎을 더 구부리세요. 무릎 각도가 너무 작습니다.")
            elif knee_angle > 120:
                print("무릎을 펴세요. 무릎 각도가 너무 큽니다.")
            
            if hip_angle < 85:
                print("엉덩이를 더 내리세요. 엉덩이가 너무 높습니다.")
            elif hip_angle > 95:
                print("엉덩이를 높이세요. 엉덩이가 너무 낮습니다.")
    
    # 실시간 영상 출력
    cv2.imshow('Pose Detection', frame)

    # 종료 조건 (Esc 키)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 종료 후 리소스 해제
cap.release()
cv2.destroyAllWindows()

