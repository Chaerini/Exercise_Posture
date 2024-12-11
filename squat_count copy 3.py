import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# MediaPipe Pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# MediaPipe Pose Connection 설정
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 모델 로드
model = tf.keras.models.load_model('exercise_model2.h5')

# 배경 이미지 로드
pullup_image = cv2.imread('pull-up-count.jpg')

# 비디오 파일 경로 설정
video_path = r'C:\Users\user\Desktop\prj_Exercise-Posture\Videos\sqart1.mp4'

# OpenCV 비디오 캡처 설정
cap = cv2.VideoCapture(video_path)

# 각도 계산 함수
def calculate_angle(point1, point2, point3):
    a = np.array(point1)  # 첫 번째 점
    b = np.array(point2)  # 두 번째 점 (기준점)
    c = np.array(point3)  # 세 번째 점
    
    # 벡터 계산
    ba = a - b
    bc = c - b
    
    # 벡터 사이 각도 계산 (라디안 -> 도)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# 기준선 저장 변수
baseline = None

# 좌표 변환 함수
def get_pixel_coordinates(landmark):
    return (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))

def set_baseline_for_exercise(exercise, landmarks, frame):
    global baseline

    if exercise == 'Squat':
        left_hip = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_HIP])
        left_knee = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_KNEE])
        distance = left_knee[1] - left_hip[1]
        if distance <= 50 and baseline is None:
            baseline = ((left_hip[0] + left_knee[0]) // 2, (left_hip[1] + left_knee[1]) // 2)
            print(f"Squat 기준선 설정됨: {baseline}")

    elif exercise == 'Push-up':
        left_elbow = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW])
        left_shoulder = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER])
        left_wrist = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_WRIST])
        elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        if elbow_angle <= 90 and baseline is None:
            baseline = (left_elbow[0], left_elbow[1])
            print(f"Push-up 기준선 설정됨: {baseline}")

    elif exercise == 'Pull-up':
        left_elbow = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW])
        left_shoulder = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER])
        left_wrist = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_WRIST])
        right_elbow = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW])
        right_wrist = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST])
        head = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.NOSE])  # 머리 랜드마크
        
        # 팔꿈치 각도 계산
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        # 조건: 양 손이 머리 위에 있고, 팔꿈치 각도가 100도 이상
        if left_wrist[1] < head[1] and right_wrist[1] < head[1] and left_elbow_angle >= 160 and right_elbow_angle >= 100:
            if baseline is None:
                baseline = ((left_wrist[0] + right_wrist[0]) / 2, (left_wrist[1] + right_wrist[1]) / 2)
                print(f"Pull-up 기준선 설정됨: {baseline}")

    # 기준선 표시
    if baseline is not None:
        x1, y1 = int(baseline[0] - 300), int(baseline[1])
        x2, y2 = int(baseline[0] + 300), int(baseline[1])
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)

# 두 점 간의 거리 계산 함수
def calculate_distance(point1, point2):
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

# 자세 판단
squat_count = 0
squat_bad_count = 0

# 상태 변수 초기화
posture_state = '스쿼트 업'
change_state = False

# 스쿼트 카운트 함수
def count_squat(left_hip, left_knee):
    global squat_count, squat_bad_count, posture_state, change_state, baseline  # 전역 변수 사용

    if baseline is not None:
        # y좌표 기준 위에서 아래로 내려갔고, 상태가 스쿼트 업일 경우
        if left_hip[1] >= baseline[1] - 20 and left_knee[1] >= baseline[1] and posture_state == '스쿼트 업':
            posture_state = '스쿼트 다운'
            change_state = True
            squat_count += 1  # 스쿼트 카운트 증가
            print(f'상태: {posture_state} 무릎y좌표: {left_knee[1]} 허리 y좌표: {left_hip[1]}')
        
        # y좌표 기준 아래에서 위로 올라갔고, 상태가 스쿼트 다운일 경우
        elif left_hip[1] < baseline[1] - 20 and left_knee[1] < baseline[1] and change_state == True:
            posture_state = '스쿼트 업'
            change_state = False
            print(f'상태: {posture_state} 무릎y좌표: {left_knee[1]} 허리 y좌표: {left_hip[1]}')

# 푸시업 카운트 함수
pushup_count = 0
pushup_state = '푸시업'  # 초기 상태: 위로 올라간 상태

def count_pushup(right_shoulder):
    global pushup_count, pushup_state, change_state
    
    print(change_state)
    if baseline is not None:
        if right_shoulder[1] >= baseline[1] and pushup_state == '푸시업' and change_state == False:  # 기준선 아래로
            pushup_state = '푸시 다운'
            change_state = True
            print(f'푸시업 왼쪽 어깨 좌표:{right_shoulder[1]} base:{baseline[1]}')
        elif right_shoulder[1] < baseline[1] and pushup_state == '푸시 다운' and change_state == True:  # 기준선 위로
            pushup_state = '푸시업'
            pushup_count += 1  # 카운트 증가
            change_state = False
            print(f"Push-up 카운트: {pushup_count}")

# 풀업 카운트 함수
pullup_count = 0
pullup_posture_state = '풀다운'  # 초기 상태: 아래로 내려간 상태

def count_pullup(head):
    global pullup_count, pullup_posture_state, change_state

    if baseline is not None:
        if head[1] <= baseline[1] and pullup_posture_state == '풀다운':  # 기준선 위로
            pullup_posture_state = '풀업'
            change_state = True
        elif head[1] > baseline[1] and pullup_posture_state == '풀업' and change_state == True:  # 기준선 아래로
            pullup_posture_state = '풀다운'
            pullup_count += 1  # 카운트 증가
            change_state = False
            print(f"Pull-up 카운트: {pullup_count}")

# 운동 분류 함수
def classify_exercise(frame):

    # 입력 사이즈에 맞게 조정
    frame_resized = cv2.resize(frame, (255, 255))
    image_normalized = frame_resized / 255.0  # 정규화
    image_input = np.expand_dims(image_normalized, axis=0)  # 차원 확장
    
    # 예측
    prediction = model.predict(image_input)

    # 예측 결과에 따라 운동 분류
    exercise_class = np.argmax(prediction, axis=1)

    if exercise_class == 0:
        exercise = 'Pull-up'
    elif exercise_class == 1:
        exercise = 'Push-up'
    elif exercise_class == 2:
        exercise = 'Squat'

    cv2.putText(frame, f'Exercise: {exercise}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    return exercise

# 좌표 변환 함수 (정규화 -> 픽셀)
def get_pixel_coordinates(landmark):
    return (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))

# 운동 클래스 변경 시 기준선과 카운트 초기화
last_exercise = None  # 마지막 운동 클래스

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 운동 분류
    exercise = classify_exercise(frame)

    # 운동 클래스가 변경된 경우 기준선과 카운트 초기화
    if exercise != last_exercise:
        last_exercise = exercise
        baseline = None  # 기준선 초기화
        squat_count = 0  # 스쿼트 카운트 초기화
        pushup_count = 0  # 푸시업 카운트 초기화
        pullup_count = 0  # 풀업 카운트 초기화
        pullup_posture_state = '풀다운'  # 풀업 초기 상태
        pushup_posture_state = '푸시업'
        posture_state = '스쿼트 업'
        change_state = False


    # 색상 변환: BGR -> RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 기준선이 설정되었으면 화면에 빨간 선으로 표시
    if baseline:
        cv2.line(frame, left_hip, left_knee, (0, 0, 255), 2)  # 빨간색 선, 두께 2
        # cv2.circle(frame, baseline, 5, (0, 255, 0), -1)  # 기준선 표시 (초록색 원)

    # Pose 모델 추적
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # 랜드마크 좌표 추출 및 변환 (허리, 무릎, 발목)
        left_hip = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_HIP])
        left_knee = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_KNEE])
        left_ankle = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE])
        left_shoulder = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER])
        right_shoulder = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER])
        head = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.NOSE])

        # 운동별 기준선 설정
        set_baseline_for_exercise(exercise, landmarks, frame)

        if exercise == 'Push-up':
            if baseline is not None:
                print(f'왼쪽 어깨: {left_shoulder[1]}, 기준선: {baseline[1]}')
            count_pushup(right_shoulder)  
            cv2.putText(frame, f'Push-up Count: {pushup_count}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif exercise == 'Pull-up':
            count_pullup(head)
            cv2.putText(frame, f'Pull-up Count: {pullup_count}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif exercise == 'Squat':
            count_squat(left_hip, left_knee)
            cv2.putText(frame, f'Squat Count: {squat_count}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 랜드마크와 관절 연결선 그리기
        mp_drawing.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        )

    # 실시간 영상 출력
    cv2.imshow('Pose Detection', cv2.resize(frame, (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5))))
    #cv2.imshow('Pose Detection', frame)

    # 종료 조건 (Esc 키)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 종료 후 리소스 해제
cap.release()
cv2.destroyAllWindows()
