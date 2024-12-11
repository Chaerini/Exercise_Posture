import cv2
import numpy as np
from roboflow import Roboflow

# Roboflow API 키로 초기화
rf = Roboflow(api_key="s1kHuWFsiF8LLbWRhXmn")
project = rf.workspace().project("exercise-pose2")  # 프로젝트 이름 입력
model = project.version(1).model  # 모델 버전 1 사용

# 동영상 파일 열기
cap = cv2.VideoCapture(r"Videos\sqart1.mp4")  # 동영상 파일 경로 입력

# 동영상이 열렸는지 확인
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()  # 한 프레임 읽기
    if not ret:
        break  # 동영상 끝나면 종료

    # 프레임 전처리 (모델이 요구하는 크기로 리사이즈하고 정규화)
    image_resized = cv2.resize(frame, (255, 255))  # 예시 크기
    image_normalized = image_resized / 255.0  # 정규화
    image_input = np.expand_dims(image_normalized, axis=0)  # 배치 차원 추가

    # 예측
    prediction = model.predict(image_input)
    
    # 예측 결과 출력
    print("예측 결과:", prediction.json())

    # 예측된 클래스를 화면에 표시
    predicted_class = prediction.json()['predictions'][0]['class']
    confidence = prediction.json()['predictions'][0]['confidence']
    
    # 프레임에 예측 정보 텍스트로 추가
    cv2.putText(frame, f'{predicted_class} ({confidence*100:.2f}%)', 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 프레임 출력
    cv2.imshow("Frame", frame)

    # 'q' 키를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 동영상 파일을 다 본 후 리소스 해제
cap.release()
cv2.destroyAllWindows()
