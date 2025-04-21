import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# مسیرها و پارامترها
data_dir = '/kaggle/input/hardfakevsrealfaces'
base_model_weights = '/kaggle/input/resnet50-w/resnet50_weights.h5'
teacher_dir = 'teacher_dir'
if not os.path.exists(teacher_dir):
    os.makedirs(teacher_dir)

img_height, img_width = 224, 224
batch_size = 32
epochs = 50

# بارگذاری و آماده‌سازی داده‌ها
df = pd.read_csv(os.path.join(data_dir, 'data.csv'))
print(df.columns)  # برای دیباگ
print(df.head())   # برای دیباگ
print(df['label'].isnull().sum())  # بررسی NaN‌ها
print(df['label'].value_counts())  # بررسی توزیع برچسب‌ها

# مدیریت NaN
df = df.dropna(subset=['label'])  # حذف ردیف‌های با label NaN

# ایجاد مسیر تصاویر (تأیید کنید 'images_id' درست است)
df['image_path'] = df['images_id'].apply(lambda x: os.path.join(data_dir, 'images', x))

# بدون نگاشت به اعداد، از برچسب‌های رشته‌ای استفاده می‌کنیم
# df['label'] = df['label'].map({'fake': 0, 'real': 1})  # این خط حذف شده

train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['label'])
train_df, val_df = train_test_split(train_val_df, test_size=0.15 / (1 - 0.15), random_state=42, stratify=train_val_df['label'])

# ژنراتورهای داده
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input
)
val_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=tf.keras.applications.resnet50.preprocess_input)
test_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=tf.keras.applications.resnet50.preprocess_input)

train_generator = train_datagen.flow_from_dataframe(train_df, x_col='image_path', y_col='label', target_size=(img_height, img_width), batch_size=batch_size, class_mode='binary')
val_generator = val_datagen.flow_from_dataframe(val_df, x_col='image_path', y_col='label', target_size=(img_height, img_width), batch_size=batch_size, class_mode='binary')
test_generator = test_datagen.flow_from_dataframe(test_df, x_col='image_path', y_col='label', target_size=(img_height, img_width), batch_size=batch_size, class_mode='binary')

# ساخت مدل
base_model = ResNet50(weights=None, include_top=False, input_shape=(img_height, img_width, 3))
base_model.load_weights(base_model_weights)  # یا weights='imagenet'
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# کامپایل
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# آموزش
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_df) // batch_size,
    validation_data=val_generator,
    validation_steps=len(val_df) // batch_size,
    epochs=epochs,
    callbacks=[early_stopping]
)

# ارزیابی
loss, accuracy = model.evaluate(test_generator)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

# ذخیره
model.save(os.path.join(teacher_dir, 'teacher_model.h5'))
