import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train_model():
    # 데이터 로드 (예시: CSV 파일로부터 데이터 불러오기)
    data = pd.read_csv(".csv")  # 예시 데이터 경로

    # feature와 label 설정
    X = data.drop('label', axis=1)  # 'label' 컬럼을 제외한 나머지 컬럼을 feature로 사용
    y = data['label']  # 'label' 컬럼을 목표 변수로 사용

    # 훈련 데이터와 테스트 데이터로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 모델 초기화 및 학습
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 예측 및 정확도 평가
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # 학습된 모델을 파일로 저장
    joblib.dump(model, 'exercise_model.pkl')

if __name__ == "__main__":
    train_model()
