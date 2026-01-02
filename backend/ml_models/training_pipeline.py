"""
Complete Training Pipeline for Medical Image Analysis Models
Handles data preparation, training, validation, and evaluation
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from xray_cnn_model import XRayCNNModel
from mri_cnn_model import MRICNNModel

class TrainingPipeline:
    """
    Complete training pipeline for medical image classification
    """
    
    def __init__(self, model_type='xray', input_shape=(150, 150, 3)):
        """
        Initialize training pipeline
        
        Args:
            model_type: 'xray' or 'mri'
            input_shape: Input image dimensions
        """
        self.model_type = model_type
        self.input_shape = input_shape
        
        if model_type == 'xray':
            self.model_class = XRayCNNModel(input_shape)
        else:
            self.model_class = MRICNNModel(input_shape)
        
        self.model = None
        self.history = None
        
    def prepare_data(self, data_dir, batch_size=32, validation_split=0.2):
        """
        Prepare data generators with augmentation
        
        Expected directory structure:
        data_dir/
            train/
                NORMAL/
                ABNORMAL/
            val/
                NORMAL/
                ABNORMAL/
            test/
                NORMAL/
                ABNORMAL/
        """
        print(f"📂 Preparing data from {data_dir}...")
        
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')
        test_dir = os.path.join(data_dir, 'test')
        
        self.train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='binary',
            shuffle=True
        )
        
        self.val_generator = test_datagen.flow_from_directory(
            val_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        self.test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        print(f"✅ Data prepared:")
        print(f"   Training samples: {self.train_generator.samples}")
        print(f"   Validation samples: {self.val_generator.samples}")
        print(f"   Test samples: {self.test_generator.samples}")
        
        return self.train_generator, self.val_generator, self.test_generator
    
    def build_and_compile(self, learning_rate=0.0001):
        """
        Build and compile the model
        """
        print("🏗️  Building model architecture...")
        self.model = self.model_class.build_model()
        self.model_class.compile_model(learning_rate=learning_rate)
        
        print("✅ Model compiled successfully!")
        print(f"   Total parameters: {self.model.count_params():,}")
        
        return self.model
    
    def setup_callbacks(self, weights_path=None, log_dir=None):
        """
        Setup training callbacks
        """
        if weights_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            weights_path = f'weights/{self.model_type}_best_{timestamp}.hdf5'
        
        if log_dir is None:
            log_dir = f'logs/{self.model_type}/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        callbacks = [
        
            ModelCheckpoint(
                filepath=weights_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=True,
                mode='max',
                verbose=1
            ),
        
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=2,
                verbose=1,
                mode='min',
                min_lr=1e-7
            ),
            

            EarlyStopping(
                monitor='val_loss',
                patience=5,
                verbose=1,
                mode='min',
                restore_best_weights=True
            ),
            
            TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        ]
        
        print(f"✅ Callbacks configured:")
        print(f"   Weights: {weights_path}")
        print(f"   Logs: {log_dir}")
        
        return callbacks
    
    def train(self, epochs=10, callbacks=None):
        """
        Train the model
        """
        print(f"🚀 Starting training for {epochs} epochs...")
        
        if callbacks is None:
            callbacks = self.setup_callbacks()
        
        self.history = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.train_generator.samples // self.train_generator.batch_size,
            epochs=epochs,
            validation_data=self.val_generator,
            validation_steps=self.val_generator.samples // self.val_generator.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Training completed!")
        
        return self.history
    
    def evaluate(self):
        """
        Evaluate model on test set
        """
        print("📊 Evaluating model on test set...")
        
        results = self.model.evaluate(
            self.test_generator,
            steps=self.test_generator.samples // self.test_generator.batch_size,
            verbose=1
        )
        
        metrics = dict(zip(self.model.metrics_names, results))
        
        print("\n✅ Evaluation Results:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value:.4f}")
        
        return metrics
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history
        """
        if self.history is None:
            print("❌ No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        

        axes[0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(self.history.history['loss'], label='Training Loss')
        axes[1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Training history saved to {save_path}")
        
        plt.show()
    
    def save_model(self, save_path):
        """
        Save complete model
        """
        self.model.save(save_path)
        print(f"✅ Model saved to {save_path}")
    
    def generate_classification_report(self):
        """
        Generate detailed classification report
        """
        from sklearn.metrics import classification_report, confusion_matrix
        import seaborn as sns
        
        print("📊 Generating classification report...")
        
        y_true = self.test_generator.classes
        y_pred_proba = self.model.predict(self.test_generator)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
      
        print("\n📄 Classification Report:")
        print(classification_report(y_true, y_pred, target_names=['Normal', 'Abnormal']))
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Abnormal'],
                   yticklabels=['Normal', 'Abnormal'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        return y_true, y_pred

def main():
    print("🏥 Documed Medical Image Analysis - Training Pipeline")
    print("="*60)
    MODEL_TYPE = 'xray'  
    DATA_DIR = './data/chest_xray'  
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.0001
    pipeline = TrainingPipeline(model_type=MODEL_TYPE)
    train_gen, val_gen, test_gen = pipeline.prepare_data(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE
    )
    model = pipeline.build_and_compile(learning_rate=LEARNING_RATE)
    model.summary()
    callbacks = pipeline.setup_callbacks()
    history = pipeline.train(epochs=EPOCHS, callbacks=callbacks)
    metrics = pipeline.evaluate()
    pipeline.plot_training_history(save_path=f'training_history_{MODEL_TYPE}.png')
    pipeline.generate_classification_report()
    pipeline.save_model(f'models/{MODEL_TYPE}_complete_model.h5')
    print("\n✅ Training pipeline completed successfully!")
    print(f"   Final Test Accuracy: {metrics['accuracy']*100:.2f}%")

if __name__ == "__main__":
    main()
