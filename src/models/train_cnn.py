"""
CNN Model Training Script for ISL Gesture Recognition
Trains a Convolutional Neural Network on gesture landmark data
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List
import pickle
import json
from datetime import datetime

class CNNTrainer:
    """CNN model trainer for gesture recognition"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize CNN trainer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Model parameters
        self.input_shape = (63,)  # 21 landmarks * 3 coordinates
        self.num_classes = 36     # 26 letters + 10 numbers
        self.batch_size = 32
        self.epochs = 100
        self.validation_split = 0.2
        self.test_split = 0.1
        
        # Data
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # Model and preprocessing
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Training history
        self.history = None
        
        # Paths
        self.data_path = self.config.get('data_path', 'data/processed/landmarks.csv')
        self.model_save_path = self.config.get('model_path', 'models/trained/cnn_model.h5')
        
        print("✅ CNN Trainer initialized")
    
    def load_data(self, data_path: str = None) -> bool:
        """
        Load training data from CSV file
        
        Args:
            data_path: Path to data file
            
        Returns:
            Success status
        """
        if data_path:
            self.data_path = data_path
        
        try:
            if not os.path.exists(self.data_path):
                print(f"❌ Data file not found: {self.data_path}")
                return False
            
            print(f"📂 Loading data from {self.data_path}")
            data = pd.read_csv(self.data_path)
            
            # Assume last column is the label, rest are features
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values
            
            print(f"✅ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
            print(f"📊 Classes: {len(np.unique(y))}")
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Split data
            X_temp, self.X_test, y_temp, self.y_test = train_test_split(
                X, y_encoded, test_size=self.test_split, random_state=42, stratify=y_encoded
            )
            
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X_temp, y_temp, test_size=self.validation_split, random_state=42, stratify=y_temp
            )
            
            # Scale features
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_val = self.scaler.transform(self.X_val)
            self.X_test = self.scaler.transform(self.X_test)
            
            # Convert to categorical
            self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
            self.y_val = keras.utils.to_categorical(self.y_val, self.num_classes)
            self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)
            
            print(f"📈 Training set: {self.X_train.shape}")
            print(f"📊 Validation set: {self.X_val.shape}")
            print(f"🧪 Test set: {self.X_test.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def create_model(self) -> keras.Model:
        """
        Create CNN model architecture
        
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            # Input layer
            keras.layers.Dense(256, activation='relu', input_shape=self.input_shape),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            # Hidden layers
            keras.layers.Dense(128, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(64, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            
            # Output layer
            keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        print("🏗️ Model created:")
        model.summary()
        
        return model
    
    def train_model(self) -> Dict:
        """
        Train the CNN model
        
        Returns:
            Training history dictionary
        """
        if self.X_train is None:
            print("❌ No training data loaded. Please load data first.")
            return {}
        
        print("🚀 Starting model training...")
        
        # Create model
        self.model = self.create_model()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath='models/checkpoints/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        start_time = datetime.now()
        
        self.history = self.model.fit(
            self.X_train, self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(self.X_val, self.y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        end_time = datetime.now()
        training_time = end_time - start_time
        
        print(f"✅ Training completed in {training_time}")
        print(f"📈 Best validation accuracy: {max(self.history.history['val_accuracy']):.4f}")
        
        return self.history.history
    
    def evaluate_model(self) -> Dict:
        """
        Evaluate model on test set
        
        Returns:
            Evaluation metrics dictionary
        """
        if self.model is None or self.X_test is None:
            print("❌ Model or test data not available")
            return {}
        
        print("📊 Evaluating model on test set...")
        
        # Test set evaluation
        test_loss, test_accuracy, test_top_k = self.model.evaluate(
            self.X_test, self.y_test, verbose=0
        )
        
        # Predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(self.y_test, axis=1)
        
        # Classification report
        class_names = self.label_encoder.classes_
        report = classification_report(
            y_true_classes, y_pred_classes,
            target_names=class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        evaluation_results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_top_k_accuracy': test_top_k,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'class_names': class_names.tolist()
        }
        
        print(f"🎯 Test Accuracy: {test_accuracy:.4f}")
        print(f"🎯 Test Top-K Accuracy: {test_top_k:.4f}")
        
        return evaluation_results
    
    def plot_training_history(self, save_path: str = None):
        """
        Plot training history
        
        Args:
            save_path: Path to save plot
        """
        if self.history is None:
            print("❌ No training history available")
            return
        
        plt.figure(figsize=(15, 5))
        
        # Accuracy plot
        plt.subplot(1, 3, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Loss plot
        plt.subplot(1, 3, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Top-K Accuracy plot
        plt.subplot(1, 3, 3)
        plt.plot(self.history.history['top_k_categorical_accuracy'], label='Training Top-K')
        plt.plot(self.history.history['val_top_k_categorical_accuracy'], label='Validation Top-K')
        plt.title('Top-K Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training plots saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], save_path: str = None):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            class_names: Class names
            save_path: Path to save plot
        """
        plt.figure(figsize=(12, 10))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def save_model(self, model_path: str = None):
        """
        Save trained model and preprocessing objects
        
        Args:
            model_path: Path to save model
        """
        if self.model is None:
            print("❌ No model to save")
            return
        
        if model_path:
            self.model_save_path = model_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        
        # Save model
        self.model.save(self.model_save_path)
        print(f"💾 Model saved to {self.model_save_path}")
        
        # Save preprocessing objects
        scaler_path = self.model_save_path.replace('.h5', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"💾 Scaler saved to {scaler_path}")
        
        # Save label encoder
        labels_path = self.model_save_path.replace('.h5', '_labels.pkl')
        with open(labels_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"💾 Labels saved to {labels_path}")
        
        # Save training configuration
        config_path = self.model_save_path.replace('.h5', '_config.json')
        config_data = {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'class_names': self.label_encoder.classes_.tolist(),
            'training_date': datetime.now().isoformat()
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        print(f"💾 Config saved to {config_path}")
    
    def generate_sample_data(self, num_samples: int = 1000):
        """
        Generate sample training data for testing
        
        Args:
            num_samples: Number of samples to generate
        """
        print(f"🎲 Generating {num_samples} sample data points...")
        
        # Generate random landmark data
        X = np.random.randn(num_samples, 63)  # 21 landmarks * 3 coordinates
        
        # Generate random labels
        gesture_classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + [str(i) for i in range(10)]
        y = np.random.choice(gesture_classes, size=num_samples)
        
        # Create DataFrame
        columns = [f'landmark_{i}_{coord}' for i in range(21) for coord in ['x', 'y', 'z']]
        columns.append('gesture')
        
        data = np.column_stack([X, y])
        df = pd.DataFrame(data, columns=columns)
        
        # Save to CSV
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        df.to_csv(self.data_path, index=False)
        
        print(f"💾 Sample data saved to {self.data_path}")
        
        return df

def main():
    """Main training function"""
    print("🚀 Starting ISL Gesture Recognition CNN Training")
    
    # Configuration
    config = {
        'data_path': 'data/processed/landmarks.csv',
        'model_path': 'models/trained/cnn_model.h5'
    }
    
    # Initialize trainer
    trainer = CNNTrainer(config)
    
    # Generate sample data if no data exists
    if not os.path.exists(config['data_path']):
        print("📝 No training data found. Generating sample data...")
        trainer.generate_sample_data(2000)
    
    # Load data
    if not trainer.load_data():
        print("❌ Failed to load data. Exiting.")
        return
    
    # Train model
    history = trainer.train_model()
    
    if not history:
        print("❌ Training failed. Exiting.")
        return
    
    # Evaluate model
    results = trainer.evaluate_model()
    
    # Plot results
    trainer.plot_training_history('models/plots/training_history.png')
    
    if 'confusion_matrix' in results:
        trainer.plot_confusion_matrix(
            np.array(results['confusion_matrix']),
            results['class_names'],
            'models/plots/confusion_matrix.png'
        )
    
    # Save model
    trainer.save_model()
    
    # Save evaluation results
    results_path = 'models/evaluation_results.json'
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"📊 Evaluation results saved to {results_path}")
    
    print("✅ Training completed successfully!")

if __name__ == "__main__":
    main()