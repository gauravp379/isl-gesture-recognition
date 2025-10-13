"""
Gesture Recognition Module for ISL System
Uses MediaPipe for hand tracking and deep learning models for classification
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Optional, Tuple
import tensorflow as tf
from tensorflow import keras
import pickle
import os
import time

class GestureRecognizer:
    """Main class for ISL gesture recognition"""
    
    def __init__(self, model_path: str = None, config: Dict = None):
        """
        Initialize the gesture recognizer
        
        Args:
            model_path: Path to trained model
            config: Configuration dictionary
        """
        self.config = config or {}
        self.model_path = model_path
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize hand tracking
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config.get('MAX_NUM_HANDS', 2),
            min_detection_confidence=self.config.get('HAND_DETECTION_CONFIDENCE', 0.7),
            min_tracking_confidence=self.config.get('HAND_TRACKING_CONFIDENCE', 0.5)
        )
        
        # Model variables
        self.model = None
        self.class_labels = None
        self.scaler = None
        
        # Performance tracking
        self.prediction_times = []
        self.frame_buffer = []
        self.max_buffer_size = 10
        
        # Load model if path provided
        if self.model_path and os.path.exists(self.model_path):
            self.load_model()
        else:
            print("⚠️ Model not found. Creating placeholder model...")
            self.create_placeholder_model()
    
    def create_placeholder_model(self):
        """Create a placeholder model for testing purposes"""
        # Create a simple CNN model structure for demonstration
        self.model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(63,)),  # 21 landmarks * 3 coordinates
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(36, activation='softmax')  # 26 letters + 10 numbers
        ])
        
        # Compile the model
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create class labels
        self.class_labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + \
                           [str(i) for i in range(10)]
        
        print("📝 Placeholder model created with {} classes".format(len(self.class_labels)))
    
    def load_model(self):
        """Load trained model from file"""
        try:
            self.model = keras.models.load_model(self.model_path)
            print(f"✅ Model loaded from {self.model_path}")
            
            # Load class labels if available
            label_path = self.model_path.replace('.h5', '_labels.pkl')
            if os.path.exists(label_path):
                with open(label_path, 'rb') as f:
                    label_encoder = pickle.load(f)
                    # Handle both LabelEncoder object and list of classes
                    if hasattr(label_encoder, 'classes_'):
                        self.class_labels = label_encoder.classes_.tolist()
                    else:
                        self.class_labels = label_encoder
            
            # Load scaler if available
            scaler_path = self.model_path.replace('.h5', '_scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                    
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.create_placeholder_model()
    
    def extract_landmarks(self, image: np.ndarray) -> Optional[Dict]:
        """
        Extract hand landmarks from image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary containing landmarks or None
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        
        # Process the image
        results = self.hands.process(rgb_image)
        
        if results.multi_hand_landmarks:
            landmarks_data = {
                'landmarks': [],
                'handedness': [],
                'bbox': [],
                'hand_count': len(results.multi_hand_landmarks)
            }
            
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Extract landmark coordinates
                landmark_list = []
                x_coords = []
                y_coords = []
                
                for landmark in hand_landmarks.landmark:
                    x = landmark.x
                    y = landmark.y
                    z = landmark.z
                    
                    landmark_list.extend([x, y, z])
                    x_coords.append(x)
                    y_coords.append(y)
                
                landmarks_data['landmarks'].append(landmark_list)
                
                # Calculate bounding box
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                landmarks_data['bbox'].append([min_x, min_y, max_x, max_y])
                
                # Get handedness (left/right)
                if results.multi_handedness:
                    handedness = results.multi_handedness[idx].classification[0].label
                    landmarks_data['handedness'].append(handedness)
            
            return landmarks_data
        
        return None
    
    def preprocess_landmarks(self, landmarks_data: Dict) -> np.ndarray:
        """
        Preprocess landmarks for model input

        Args:
            landmarks_data: Raw landmarks data

        Returns:
            Preprocessed feature vector
        """
        if not landmarks_data or not landmarks_data['landmarks']:
            return np.zeros((63,))  # 21 landmarks * 3 coordinates

        # Process both hands if available, use the one with better visibility
        best_features = None
        best_score = -1

        for hand_idx, landmarks in enumerate(landmarks_data['landmarks']):
            # Convert to numpy array
            features = np.array(landmarks, dtype=np.float32)

            # Calculate a simple score based on landmark spread (better hand detection)
            wrist = features[:3]  # First 3 values are wrist x,y,z
            distances = np.linalg.norm(features.reshape(-1, 3) - wrist.reshape(1, 3), axis=1)
            score = np.mean(distances)  # Higher score = better hand pose

            # Normalize landmarks
            features = self.normalize_landmarks(features)

            # Apply scaler if available
            if self.scaler:
                features = self.scaler.transform(features.reshape(1, -1)).flatten()

            # Keep the hand with the best score
            if score > best_score:
                best_score = score
                best_features = features

        return best_features if best_features is not None else np.zeros((63,))
    
    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmark coordinates
        
        Args:
            landmarks: Raw landmark coordinates
            
        Returns:
            Normalized landmarks
        """
        # Reshape to (21, 3) for processing
        landmarks_2d = landmarks.reshape(-1, 3)
        
        # Get wrist landmark (index 0) as reference point
        wrist = landmarks_2d[0]
        
        # Translate to origin (wrist = [0, 0, 0])
        landmarks_2d = landmarks_2d - wrist
        
        # Scale to unit size
        max_distance = np.max(np.linalg.norm(landmarks_2d, axis=1))
        if max_distance > 0:
            landmarks_2d = landmarks_2d / max_distance
        
        return landmarks_2d.flatten()
    
    def predict(self, image: np.ndarray) -> Optional[Dict]:
        """
        Predict gesture from image
        
        Args:
            image: Input image
            
        Returns:
            Prediction results or None
        """
        start_time = time.time()
        
        try:
            # Extract landmarks
            landmarks_data = self.extract_landmarks(image)
            
            if landmarks_data is None:
                return None
            
            # Preprocess features
            features = self.preprocess_landmarks(landmarks_data)
            
            if self.model is None:
                return None
            
            # Make prediction
            features_batch = features.reshape(1, -1)
            predictions = self.model.predict(features_batch, verbose=0)
            
            # Get the most confident prediction
            confidence = float(np.max(predictions[0]))
            predicted_class_idx = int(np.argmax(predictions[0]))
            
            if self.class_labels and predicted_class_idx < len(self.class_labels):
                gesture = self.class_labels[predicted_class_idx]
            else:
                gesture = f"Class_{predicted_class_idx}"
            
            # Calculate prediction time
            prediction_time = time.time() - start_time
            self.prediction_times.append(prediction_time)
            
            # Keep only last 100 prediction times
            if len(self.prediction_times) > 100:
                self.prediction_times.pop(0)
            
            return {
                'gesture': gesture,
                'confidence': confidence,
                'landmarks_data': landmarks_data,
                'prediction_time': prediction_time,
                'hand_count': landmarks_data['hand_count']
            }
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None
    
    def draw_landmarks(self, image: np.ndarray, landmarks_data: Dict = None) -> np.ndarray:
        """
        Draw hand landmarks on image
        
        Args:
            image: Input image
            landmarks_data: Pre-extracted landmarks (optional)
            
        Returns:
            Image with landmarks drawn
        """
        if landmarks_data is None:
            landmarks_data = self.extract_landmarks(image)
        
        if landmarks_data is None:
            return image
        
        # Convert to RGB for processing
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = True
        
        # Re-run MediaPipe to get drawing data
        results = self.hands.process(rgb_image)
        
        # Convert back to BGR for display
        annotated_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Draw bounding box
                if 'bbox' in landmarks_data and idx < len(landmarks_data['bbox']):
                    bbox = landmarks_data['bbox'][idx]
                    h, w, _ = annotated_image.shape
                    
                    x1, y1, x2, y2 = bbox
                    x1, x2 = int(x1 * w), int(x2 * w)
                    y1, y2 = int(y1 * h), int(y2 * h)
                    
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw handedness label
                    if 'handedness' in landmarks_data and idx < len(landmarks_data['handedness']):
                        handedness = landmarks_data['handedness'][idx]
                        cv2.putText(annotated_image, handedness, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated_image
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model
        
        Returns:
            Model information dictionary
        """
        if self.model is None:
            return {'error': 'No model loaded'}
        
        return {
            'model_type': 'CNN' if 'CNN' in str(type(self.model)) else 'Sequential',
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'num_parameters': self.model.count_params(),
            'num_classes': len(self.class_labels) if self.class_labels else 0,
            'class_labels': self.class_labels,
            'avg_prediction_time': np.mean(self.prediction_times) if self.prediction_times else 0,
            'model_loaded': True
        }
    
    def get_performance_metrics(self) -> Dict:
        """
        Get performance metrics
        
        Returns:
            Performance metrics dictionary
        """
        return {
            'avg_prediction_time': np.mean(self.prediction_times) if self.prediction_times else 0,
            'min_prediction_time': np.min(self.prediction_times) if self.prediction_times else 0,
            'max_prediction_time': np.max(self.prediction_times) if self.prediction_times else 0,
            'total_predictions': len(self.prediction_times),
            'fps_estimate': 1.0 / np.mean(self.prediction_times) if self.prediction_times else 0
        }
    
    def save_model(self, path: str):
        """
        Save the trained model
        
        Args:
            path: Path to save the model
        """
        if self.model:
            self.model.save(path)
            
            # Save class labels
            if self.class_labels:
                label_path = path.replace('.h5', '_labels.pkl')
                with open(label_path, 'wb') as f:
                    pickle.dump(self.class_labels, f)
            
            # Save scaler
            if self.scaler:
                scaler_path = path.replace('.h5', '_scaler.pkl')
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
            
            print(f"✅ Model saved to {path}")
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'hands'):
            self.hands.close()
        print("🧹 Gesture recognizer cleaned up")