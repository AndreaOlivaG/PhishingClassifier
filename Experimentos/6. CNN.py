import os
import scipy
import random
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score, accuracy_score, classification_report
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, MaxPool2D

seed_value = 0
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = str(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

path = "Figures/"

train_gen = ImageDataGenerator(validation_split=0.2)
test_gen = ImageDataGenerator()
batch_size = 128
train_dataset = train_gen.flow_from_directory(path+'train/', target_size=(369, 369), batch_size=batch_size,
                                              class_mode='binary', subset="training", seed=43, color_mode='rgb')
validation_dataset = train_gen.flow_from_directory(path+'train/', target_size=(369, 369), batch_size=batch_size,
                                                   class_mode='binary', subset="validation", seed=43, color_mode='rgb')
test_dataset = test_gen.flow_from_directory(path+'test/', target_size=(369, 369), batch_size=batch_size,
                                            class_mode='binary', shuffle=False, color_mode='rgb')

model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(369, 369, 3)))
model.add(MaxPool2D(2, 2))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPool2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(2, 2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.001), metrics=['accuracy'])
history = model.fit(train_dataset, steps_per_epoch=5, validation_data=validation_dataset, validation_steps=5,
                    epochs=6, verbose=2)

predictions = model.predict(test_dataset).round()

test_labels = test_dataset.labels
print(classification_report(test_labels, predictions))

f1 = f1_score(test_labels, predictions, average='macro')
acc = accuracy_score(test_labels, predictions)
print("f1-score: %.4f, accuracy: %.4f" % (f1, acc))
