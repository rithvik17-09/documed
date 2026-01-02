"""
MRI CNN Model for Brain Abnormality Detection
Similar architecture to X-ray model but optimized for MRI scans
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.layers import (
    Input, Conv2D, SeparableConv2D, MaxPool2D, BatchNormalization,
    Dropout, Flatten, Dense
)
import numpy as np

class MRICNNModel:
    """
    Convolutional Neural Network for Brain MRI Analysis
    Detects Normal vs Abnormal (lesions, tumors, etc.)
    """
    
    def __init__(self, input_shape=(150, 150, 3)):
        self.input_shape = input_shape
        self.model = None
        
    def build_model(self):
        """
        Build CNN architecture for MRI analysis
        Same architecture as X-ray model for consistency
        """
        
        inputs = Input(shape=self.input_shape)
        
        # Block 1
        x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
        x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        
        # Block 2
        x = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        
        # Block 3
        x = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        
        # Block 4
        x = SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Dropout(rate=0.2)(x)
        
        # Block 5
        x = SeparableConv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = SeparableConv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Dropout(rate=0.2)(x)
        
        # Dense layers
        x = Flatten()(x)
        x = Dense(units=512, activation='relu')(x)
        x = Dropout(rate=0.7)(x)
        x = Dense(units=128, activation='relu')(x)
        x = Dropout(rate=0.5)(x)
        x = Dense(units=64, activation='relu')(x)
        x = Dropout(rate=0.3)(x)
        
        # Output
        output = Dense(units=1, activation='sigmoid')(x)
        
        self.model = models.Model(inputs=inputs, outputs=output, name='MRI_CNN')
        
        return self.model
    
    def compile_model(self):
        """
        Compile model with optimizer and loss
        """
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
    
    def predict(self, image):
        """
        Make prediction on MRI image
        """
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        prediction = self.model.predict(image)[0][0]
        classification = 'Defective' if prediction > 0.5 else 'Normal'
        confidence = prediction if prediction > 0.5 else (1 - prediction)
        
        return {
            'prediction': float(prediction),
            'classification': classification,
            'confidence': float(confidence) * 100
        }

if __name__ == "__main__":
    mri_model = MRICNNModel()
    model = mri_model.build_model()
    mri_model.compile_model()
    print("✅ MRI Model built successfully!")
