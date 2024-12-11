import tensorflow as tf

# 'data/train' 디렉터리에서 이미지 데이터를 로드하여 훈련 데이터셋 생성
train_dataset = tf.keras.utils.image_dataset_from_directory(
    'data/train',
    image_size=(255, 255),
    batch_size=32
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    'data/validation',
    image_size=(255, 255),
    batch_size=32
)