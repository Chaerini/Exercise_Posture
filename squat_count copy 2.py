import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# MediaPipe Pose Connection 설정
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 비디오 파일 경로 설정
video_path = r'C:\Users\admin\Desktop\모블_텐서플로우\prj_Exercise-Posture\Videos\pushup.mp4'

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

def set_baseline_for_exercise(exercise, landmarks, frame):
    global baseline

    # 좌표 변환 함수
    def get_pixel_coordinates(landmark):
        return (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))

    if exercise == 'Squat':
        left_hip = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_HIP])
        left_knee = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_KNEE])
        distance = left_knee[1] - left_hip[1]
        if distance <= 20 and baseline is None:
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
        elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        if elbow_angle <= 90 and baseline is None:
            baseline = (left_elbow[0], left_elbow[1])
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
def count_squat(left_hip, left_knee, waist_angle):
    global squat_count, squat_bad_count, posture_state, change_state, baseline  # 전역 변수 사용

    if baseline is not None:
        # y좌표 기준 위에서 아래로 내려갔고, 상태가 스쿼트 업일 경우
        if left_hip[1] >= baseline[1] and left_knee[1] >= baseline[1] and waist_angle <= 40 and posture_state == '스쿼트 업':
            posture_state = '스쿼트 다운'
            change_state = True
            squat_count += 1  # 스쿼트 카운트 증가
            print(f'상태: {posture_state} 무릎y좌표: {left_knee[1]} 허리 y좌표: {left_hip[1]}')
        
        # y좌표 기준 아래에서 위로 올라갔고, 상태가 스쿼트 다운일 경우
        elif left_hip[1] < baseline[1] and left_knee[1] < baseline[1] and change_state == True:
            posture_state = '스쿼트 업'
            change_state = False
            print(f'상태: {posture_state} 무릎y좌표: {left_knee[1]} 허리 y좌표: {left_hip[1]}')

# 푸시업 카운트 함수
pushup_count = 0
pushup_state = '푸시업'  # 초기 상태: 위로 올라간 상태

def count_pushup(left_shoulder):
    global pushup_count, pushup_state, change_state
    
    print(change_state)
    if baseline is not None:
        if left_shoulder[1] >= baseline[1] - 20 and pushup_state == '푸시업':  # 기준선 아래로
            pushup_state = '푸시 다운'
            change_state = True
            print(f'푸시업 왼쪽 어깨 좌표:{left_shoulder[1]} base:{baseline[1]}')
        elif left_shoulder[1] < baseline[1] and pushup_state == '푸시 다운' and change_state == True:  # 기준선 위로
            pushup_state = '푸시업'
            pushup_count += 1  # 카운트 증가
            change_state = False
            print(f"Push-up 카운트: {pushup_count}")

# 운동 분류 함수
def classify_exercise(landmarks, frame):
    left_hand = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_WRIST])
    right_hand = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST])
    face_y = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.NOSE])[1]
    foot_y = min(get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE])[1],
                 get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE])[1])

    if left_hand[1] < face_y and right_hand[1] < face_y:
        exercise = 'Pull-up'
    elif left_hand[1] > foot_y - 30 and right_hand[1] > foot_y - 30:
        exercise = 'Push-up'
    else:
        exercise = 'Squat'

    cv2.putText(frame, f'Exercise: {exercise}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    return exercise

def get_pixel_coordinates(landmark):
    return (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 색상 변환: BGR -> RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 기준선이 설정되었으면 화면에 빨간 선으로 표시
    if baseline:
        cv2.line(frame, left_hip, left_knee, (0, 0, 255), 2)  # 빨간색 선, 두께 2
        cv2.circle(frame, baseline, 5, (0, 255, 0), -1)  # 기준선 표시 (초록색 원)

    # Pose 모델 추적
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # 운동 분류
        exercise = classify_exercise(landmarks, frame)

        # 좌표 변환 함수 (정규화 -> 픽셀)
        def get_pixel_coordinates(landmark):
            return (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))

        # 랜드마크 좌표 추출 및 변환 (허리, 무릎, 발목)
        left_hip = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_HIP])
        left_knee = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_KNEE])
        left_ankle = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE])
        left_shoulder = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER])

        left_hip_90 = [left_hip[0] + 90, left_hip[1] + 90]
        # 무릎 및 허리 각도 계산
        knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        waist_angle = calculate_angle([left_hip[0], left_hip[1] - 100], left_hip, left_shoulder)

        # 90도 직선과 left_hip의 각도 차이를 구합니다.
        waist_angle = min(waist_angle, 90)  # 90도를 넘지 않도록 제한


        # 무릎 각도 표시
        cv2.putText(frame, f'Knee: {int(knee_angle)} deg', 
                    (left_knee[0], left_knee[1] - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 무릎 각도 호 그리기
        center = left_knee
        radius = 50
        start_angle = int(np.degrees(np.arctan2(left_hip[1] - left_knee[1], left_hip[0] - left_knee[0])))
        end_angle = start_angle + int(knee_angle)
        cv2.ellipse(frame, center, (radius, radius), 0, start_angle, end_angle, (0, 255, 255), 2)

        # 허리 각도 표시
        cv2.putText(frame, f'Waist: {int(waist_angle)} deg', 
                    (left_hip[0], left_hip[1] - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # 허리 각도 호 그리기
        waist_center = left_hip
        waist_radius = 50
        waist_start_angle = int(np.degrees(np.arctan2(left_shoulder[1] - left_hip[1], left_shoulder[0] - left_hip[0])))
        waist_end_angle = waist_start_angle + int(waist_angle)
        cv2.ellipse(frame, waist_center, (waist_radius, waist_radius), 0, waist_start_angle, waist_end_angle, (255, 0, 255), 2)

        
        # # 무릎에서 위로 향하는 직선 그리기
        # knee_vertical = (left_knee[0], left_knee[1] - 100)
        # cv2.line(frame, left_knee, knee_vertical, (255, 0, 0), 2)

        # 허리에서 위로 향하는 직선 그리기
        hip_vertical = (left_hip[0], left_hip[1] - 100)
        cv2.line(frame, left_hip, hip_vertical, (0, 255, 0), 2)

        # 운동별 기준선 설정
        set_baseline_for_exercise(exercise, landmarks, frame)

        if exercise == 'Push-up':
            count_pushup(left_shoulder)
            cv2.putText(frame, f'Push-up Count: {pushup_count}', (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            #cv2.putText(frame, f'sholder: {left_shoulder[1]} base{baseline[1]}', (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        #count_squat(left_hip, left_knee, waist_angle)

        # 각도 표시
        cv2.putText(frame, f'Knee Angle: {int(knee_angle)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f'Waist Angle: {int(waist_angle)}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 자세 판단 결과 표시
        #cv2.putText(frame, f'squat_count: {squat_count}/15', 
        #            (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
    # cv2.imshow('Pose Detection', frame)

    # 종료 조건 (Esc 키)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 종료 후 리소스 해제
cap.release()
cv2.destroyAllWindows()
