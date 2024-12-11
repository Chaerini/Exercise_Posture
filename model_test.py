import cv2
import numpy as np
import tensorflow as tf

# 모델 로드
model = tf.keras.models.load_model('exercise_model.h5')

# 테스트 이미지 불러오기 (예시로 한 장의 이미지를 불러옵니다)
image = cv2.imread(r'data_sqart\test\4.png')

# 이미지 전처리 (모델이 요구하는 크기로 리사이즈하고 정규화)
image_resized = cv2.resize(image, (255, 255))  # 예시 크기

cv2.imshow("w", image_resized)

# 'q' 키를 눌러 종료
cv2.waitKey(1)

image_normalized = image_resized / 255.0  # 정규화

# 차원 확장 (배치 차원 추가)
image_input = np.expand_dims(image_normalized, axis=0)

# 예측
prediction = model.predict(image_input)

# 예측 결과 출력
print("예측 결과:", prediction)

predicted_class = np.argmax(prediction, axis=1)
print(f"예측된 클래스: {predicted_class}")