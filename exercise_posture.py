import cv2
import mediapipe as mp

# MediaPipe Pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# MediaPipe Pose Connection 설정
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 비디오 파일 경로 설정
video_path = r'C:\Users\admin\Desktop\모블_텐서플로우\prj_Exercise-Posture\Videos\Squrt.mp4'

# OpenCV 비디오 캡처 설정
cap = cv2.VideoCapture(video_path)

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
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(), # 올바른 메서드 이름
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)  # 대체 스타일 지정
        )

    # 실시간 영상 출력
    cv2.imshow('Pose Detection', frame)

    # 종료 조건 (Esc 키)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 종료 후 리소스 해제
cap.release()
cv2.destroyAllWindows()
