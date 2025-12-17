import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import numpy as np

def create_dummy_model():
    print("Creating a DUMMY model for UI testing purposes (3 Classes)...")
    
    # Base model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    # Changed to 3 classes: BACTERIA, NORMAL, VIRUS
    predictions = Dense(3, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Saving dummy model to 'pneumonia_model.h5'...")
    model.save('pneumonia_model.h5')
    print("Model saved! Classes expected: 0=BACTERIA, 1=NORMAL, 2=VIRUS")

if __name__ == "__main__":
    create_dummy_model()
