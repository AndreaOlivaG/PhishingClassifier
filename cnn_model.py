import os
import random

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score, classification_report
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

seed = 43
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

path = "data/figures/"

train_gen = ImageDataGenerator(validation_split=0.2)
test_gen = ImageDataGenerator()
batch_size = 128
train_dataset = train_gen.flow_from_directory(path + 'train/', target_size=(369, 369), batch_size=batch_size,
                                              class_mode='binary', subset="training", seed=43, color_mode='rgb')
validation_dataset = train_gen.flow_from_directory(path + 'train/', target_size=(369, 369), batch_size=batch_size,
                                                   class_mode='binary', subset="validation", seed=43, color_mode='rgb')
test_dataset = test_gen.flow_from_directory(path + 'test/', target_size=(369, 369), batch_size=batch_size,
                                            class_mode='binary', shuffle=False, color_mode='rgb')

model = tf.keras.Sequential([
    tf.keras.Input(shape=(369, 369, 3)),
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.001), metrics=['accuracy'])
history = model.fit(train_dataset, steps_per_epoch=5, validation_data=validation_dataset, validation_steps=5,
                    epochs=6, verbose=2)

predictions = model.predict(test_dataset).round()
test_labels = test_dataset.labels
print(classification_report(test_labels, predictions))

f1 = f1_score(test_labels, predictions, average='macro')
acc = accuracy_score(test_labels, predictions)
print("f1-score: %.4f, accuracy: %.4f" % (f1, acc))
