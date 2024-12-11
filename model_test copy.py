import cv2
import numpy as np
import tensorflow as tf

# 모델 로드
model = tf.keras.models.load_model('exercise_model2.h5')

# 동영상 파일 경로 또는 웹캠(0)
cap = cv2.VideoCapture(r'C:\Users\user\Desktop\prj_Exercise-Posture\Videos\sqart1.mp4')  # 파일 경로 또는 '0'으로 웹캠 사용

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 이미지 전처리 (모델이 요구하는 크기로 리사이즈하고 정규화)
    image_resized = cv2.resize(frame, (255, 255))  # 모델 입력 크기
    image_normalized = image_resized / 255.0  # 정규화
    image_input = np.expand_dims(image_normalized, axis=0)  # 차원 확장
    
    # 예측
    prediction = model.predict(image_input)
    predicted_class = np.argmax(prediction, axis=1)
    
    # 예측 결과 표시
    label = f"Class: {predicted_class[0]}, Confidence: {prediction[0][predicted_class[0]]:.2f}"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 프레임 표시
    cv2.imshow('Video', frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
