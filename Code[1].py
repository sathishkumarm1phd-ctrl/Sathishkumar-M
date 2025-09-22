import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import (
    Dense, Flatten, Input, Lambda, GlobalAveragePooling2D, Reshape,
    Multiply, Conv2D, Concatenate
)
from tensorflow.keras.models import Model

def preprocess_layer():
    return Lambda(lambda x: tf.keras.applications.efficientnet.preprocess_input(
        tf.image.resize(x, (224, 224))
    ))

def extract_features(image_tensor):
    base_model = EfficientNetB0(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
    base_model.trainable = False
    return base_model(image_tensor)

def apply_channel_attention(features):
    channel = features.shape[-1]
    squeeze = GlobalAveragePooling2D()(features)
    excitation = Dense(channel // 16, activation='relu')(squeeze)
    excitation = Dense(channel, activation='sigmoid')(excitation)
    excitation = Reshape((1, 1, channel))(excitation)
    return Multiply()([features, excitation])

def apply_spatial_attention(features):
    avg_pool = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(features)
    max_pool = Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(features)
    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    attention = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)
    return Multiply()([features, attention])

def QDA_EffB0_Model():
    input_image = Input(shape=(224, 224, 3))

    x = preprocess_layer()(input_image)
    features = extract_features(x)

    channel_att = apply_channel_attention(features)
    spatial_att = apply_spatial_attention(channel_att)

    flat = Flatten()(spatial_att)
    quantum_encoded = Dense(128, activation='relu')(flat)
    output = Dense(10, activation='softmax')(quantum_encoded)

    return Model(inputs=input_image, outputs=output)

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def predict_from_folder_recursive(root_folder):
    model = QDA_EffB0_Model()
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(dirpath, file)
                try:
                    input_tensor = load_and_preprocess_image(img_path)
                    predictions = model.predict(input_tensor)
                    predicted_class = np.argmax(predictions[0])

                    class_folder = os.path.basename(os.path.dirname(img_path))
                    print(f"[{class_folder}] {file} → Predicted class index: {predicted_class}")
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    root_folder = r'path_of_the_downloaded_datset_path'  
    predict_from_folder_recursive(root_folder)
