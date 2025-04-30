from keras.models import load_model
from keras.applications import ResNet50
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

# GPU사용이 안될 경우 CPU로 학습하기 위함.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# 데이터 로드
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# ResNet50 기반 모델 생성
base_model = ResNet50(weights=None, include_top=False, input_shape=(32, 32, 3))
x = Flatten()(base_model.output)
output = Dense(10, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# 컴파일
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 훈련
model.fit(x_train, to_categorical(y_train), epochs=5, batch_size=64, validation_split=0.1)

# 저장
model.save('model_A.h5')

# 약간 변경한 Model_B 생성
model_B = Model(inputs=base_model.input, outputs=output)
model_B.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
model_B.fit(x_train, to_categorical(y_train), epochs=5, batch_size=64, validation_split=0.1)
model_B.save('model_B.h5')


model1 = load_model("model_A.h5")
model2 = load_model("model_B.h5")

# 일부 샘플만 선택
input_data = x_test[:1000]
labels = y_test[:1000]

# 탐색 index list
index_list = []
# 예측 결과 불일치 탐지
for i, sample in enumerate(input_data):
    pred1 = model1.predict(sample.reshape(1, 32, 32, 3))
    pred2 = model2.predict(sample.reshape(1, 32, 32, 3))
    if pred1.argmax() != pred2.argmax():
        index_list.append(i)

print(index_list)
print(len(index_list))