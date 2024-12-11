import cv2
import torch
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore

# YOLO 모델 로드 (PyTorch로 로드)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 'yolov5s'는 경량 모델

# 스쿼트 모델 로드 (TensorFlow 모델)
squat_model = tf.keras.models.load_model('squat.h5')

# 카운트 및 상태 변수 초기화
squat_count = 0
current_state = "up"  # 초기 상태: 스쿼트 시작 전

# 비디오 캡처 설정
video_path = r'C:\Users\admin\Desktop\모블_텐서플로우\prj_Exercise-Posture\Videos\Squrt.mp4'
cap = cv2.VideoCapture(0)

# YOLO로 사람 감지
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 모델로 객체 감지
    results = model(frame)  # YOLOv5 모델을 통해 객체 감지
    detections = results.pandas().xywh[0]  # 감지된 객체의 정보

    print(detections.columns)

    # 사람 객체만 필터링 (클래스 0: 사람)
    people = detections[detections['class'] == 0]

    if not people.empty:
        for index, row in people.iterrows():
            # 사람의 바운딩 박스 정보 (results.pandas()의 열 이름 확인)
            x_center, y_center, width, height = row['xcenter'], row['ycenter'], row['width'], row['height']
            frame_height, frame_width, _ = frame.shape

            # 바운딩 박스를 좌표로 변환
            x1 = int((x_center - width / 2) * frame_width)
            y1 = int((y_center - height / 2) * frame_height)
            x2 = int((x_center + width / 2) * frame_width)
            y2 = int((y_center + height / 2) * frame_height)

            # 사람의 바운딩 박스를 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 사람 이미지 추출 (바운딩 박스 영역)
            person_image = frame[y1:y2, x1:x2]

            print(person_image.shape)

            # 이미지 크기 조정
            person_image_resized = cv2.resize(frame, (255, 255))

            # 모델 예측을 위한 이미지 전처리 (스쿼트 모델에 맞게 전처리)
            input_data = np.expand_dims(person_image_resized, axis=0)  # 배치 차원 추가
            input_data = input_data / 255.0  # 정규화

            # 모델 예측
            prediction = squat_model.predict(input_data)
            predicted_class = np.argmax(prediction)  # 0: Bad, 1: Up, 2: Down
            squat_bad = prediction[0][0]  # 클래스 0 (잘못된 자세)의 확률
            squat_up = prediction[0][1]  # 클래스 1 (잘못된 자세)의 확률
            squat_down = prediction[0][2]  # 클래스 2 (잘못된 자세)의 확률

            print(f'예측값: {prediction}')

            # 상태 전환 및 카운트 관리
            if predicted_class == 2 and current_state == "up":
                squat_count += 1
                current_state = "down"
                print("스쿼트 카운트:", squat_count)
            elif predicted_class == 1:
                current_state = "up"
                print("스쿼트 상태: 올라감.")
            elif squat_bad >= 0.8:
                print("스쿼트 상태: 잘못된 자세")
            else:
                print("스쿼트 상태: 하는 중")

    # 화면에 카운트 표시
    cv2.putText(frame, f'Squat Count: {squat_count}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # 결과 이미지 출력
    cv2.imshow('Squat Tracker', frame)

    # 'q' 키를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
