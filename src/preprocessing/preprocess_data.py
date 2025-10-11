"""
Data Preprocessing Module for ISL Gesture Recognition
Processes raw gesture data and prepares it for model training
"""
import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from typing import List, Dict, Tuple, Optional
import json
import pickle
from datetime import datetime
import glob
from pathlib import Path

class GestureDataPreprocessor:
    """Preprocessor for gesture training data"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize data preprocessor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Data paths
        self.raw_data_path = self.config.get('raw_data_path', 'data/raw')
        self.processed_data_path = self.config.get('processed_data_path', 'data/processed')
        
        # Processing parameters
        self.image_size = (224, 224)
        self.normalize_landmarks = True
        self.augment_data = True
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'processed_images': 0,
            'failed_images': 0,
            'gestures_count': {},
            'hand_detection_rate': 0
        }
        
        print("✅ Gesture Data Preprocessor initialized")
    
    def extract_landmarks_from_image(self, image_path: str) -> Optional[Dict]:
        """
        Extract hand landmarks from single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary containing landmarks or None
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                print(f"⚠️ Could not read image: {image_path}")
                return None
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process image
            results = self.hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                landmarks_data = {
                    'image_path': image_path,
                    'landmarks': [],
                    'handedness': [],
                    'image_shape': image.shape
                }
                
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Extract landmark coordinates
                    landmark_list = []
                    
                    for landmark in hand_landmarks.landmark:
                        landmark_list.extend([landmark.x, landmark.y, landmark.z])
                    
                    landmarks_data['landmarks'].append(landmark_list)
                    
                    # Get handedness
                    if results.multi_handedness:
                        handedness = results.multi_handedness[idx].classification[0].label
                        landmarks_data['handedness'].append(handedness)
                
                return landmarks_data
            
            return None
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            return None
    
    def normalize_landmarks(self, landmarks: List[float]) -> List[float]:
        """
        Normalize landmark coordinates
        
        Args:
            landmarks: Raw landmark coordinates
            
        Returns:
            Normalized landmarks
        """
        if not landmarks:
            return landmarks
        
        # Convert to numpy array and reshape
        landmarks_array = np.array(landmarks).reshape(-1, 3)
        
        # Get wrist landmark (index 0) as reference point
        wrist = landmarks_array[0]
        
        # Translate to origin (wrist = [0, 0, 0])
        normalized = landmarks_array - wrist
        
        # Scale to unit size
        max_distance = np.max(np.linalg.norm(normalized, axis=1))
        if max_distance > 0:
            normalized = normalized / max_distance
        
        return normalized.flatten().tolist()
    
    def augment_landmarks(self, landmarks: List[float], num_augmentations: int = 3) -> List[List[float]]:
        """
        Augment landmark data
        
        Args:
            landmarks: Original landmarks
            num_augmentations: Number of augmented versions to create
            
        Returns:
            List of augmented landmark sets
        """
        if not landmarks:
            return [landmarks]
        
        augmented_data = [landmarks]  # Include original
        landmarks_array = np.array(landmarks).reshape(-1, 3)
        
        for _ in range(num_augmentations):
            augmented = landmarks_array.copy()
            
            # Add small random noise
            noise = np.random.normal(0, 0.02, augmented.shape)
            augmented += noise
            
            # Random rotation (small angle)
            angle = np.random.uniform(-0.1, 0.1)  # radians
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            
            # Apply rotation
            for i in range(len(augmented)):
                augmented[i] = rotation_matrix @ augmented[i]
            
            # Random scaling
            scale = np.random.uniform(0.95, 1.05)
            augmented *= scale
            
            augmented_data.append(augmented.flatten().tolist())
        
        return augmented_data
    
    def process_dataset_folder(self, dataset_path: str) -> pd.DataFrame:
        """
        Process entire dataset folder structure
        Expected structure: dataset_path/gesture_class/image_files
        
        Args:
            dataset_path: Path to dataset folder
            
        Returns:
            DataFrame with processed data
        """
        print(f"📂 Processing dataset: {dataset_path}")
        
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset path not found: {dataset_path}")
            return pd.DataFrame()
        
        all_data = []
        gesture_folders = [d for d in os.listdir(dataset_path) 
                          if os.path.isdir(os.path.join(dataset_path, d))]
        
        total_images = 0
        for gesture in gesture_folders:
            gesture_path = os.path.join(dataset_path, gesture)
            image_files = glob.glob(os.path.join(gesture_path, "*.jpg")) + \
                         glob.glob(os.path.join(gesture_path, "*.png")) + \
                         glob.glob(os.path.join(gesture_path, "*.jpeg"))
            total_images += len(image_files)
        
        self.stats['total_images'] = total_images
        processed_count = 0
        
        print(f"📊 Found {len(gesture_folders)} gesture classes with {total_images} total images")
        
        for gesture in gesture_folders:
            gesture_path = os.path.join(dataset_path, gesture)
            print(f"Processing gesture: {gesture}")
            
            # Find all image files
            image_files = glob.glob(os.path.join(gesture_path, "*.jpg")) + \
                         glob.glob(os.path.join(gesture_path, "*.png")) + \
                         glob.glob(os.path.join(gesture_path, "*.jpeg"))
            
            gesture_count = 0
            
            for image_file in image_files:
                # Extract landmarks
                landmarks_data = self.extract_landmarks_from_image(image_file)
                
                if landmarks_data:
                    # Process each hand detected
                    for hand_idx, landmarks in enumerate(landmarks_data['landmarks']):
                        # Normalize landmarks
                        if self.normalize_landmarks:
                            landmarks = self.normalize_landmarks(landmarks)
                        
                        # Create base data entry
                        data_entry = {
                            'image_path': image_file,
                            'gesture': gesture,
                            'hand_index': hand_idx,
                            'handedness': landmarks_data['handedness'][hand_idx] if hand_idx < len(landmarks_data['handedness']) else 'Unknown'
                        }
                        
                        # Add landmark coordinates
                        for i, coord in enumerate(landmarks):
                            data_entry[f'landmark_{i}'] = coord
                        
                        all_data.append(data_entry)
                        gesture_count += 1
                        
                        # Data augmentation
                        if self.augment_data:
                            augmented_landmarks = self.augment_landmarks(landmarks, 2)
                            
                            for aug_idx, aug_landmarks in enumerate(augmented_landmarks[1:], 1):
                                aug_entry = data_entry.copy()
                                aug_entry['image_path'] = f"{image_file}_aug_{aug_idx}"
                                
                                # Update landmark coordinates
                                for i, coord in enumerate(aug_landmarks):
                                    aug_entry[f'landmark_{i}'] = coord
                                
                                all_data.append(aug_entry)
                                gesture_count += 1
                    
                    processed_count += 1
                else:
                    self.stats['failed_images'] += 1
                
                # Progress update
                if processed_count % 50 == 0:
                    print(f"  Processed {processed_count}/{total_images} images...")
            
            self.stats['gestures_count'][gesture] = gesture_count
            print(f"  {gesture}: {gesture_count} samples generated")
        
        self.stats['processed_images'] = processed_count
        self.stats['hand_detection_rate'] = processed_count / total_images if total_images > 0 else 0
        
        print(f"✅ Processing completed: {processed_count}/{total_images} images processed")
        print(f"📊 Hand detection rate: {self.stats['hand_detection_rate']:.2%}")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        if not df.empty:
            print(f"📋 Generated dataset shape: {df.shape}")
            print(f"📊 Gesture distribution:")
            print(df['gesture'].value_counts())
        
        return df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = 'landmarks.csv'):
        """
        Save processed data to CSV file
        
        Args:
            df: Processed DataFrame
            filename: Output filename
        """
        if df.empty:
            print("⚠️ No data to save")
            return
        
        # Ensure output directory exists
        os.makedirs(self.processed_data_path, exist_ok=True)
        
        # Save main dataset
        output_path = os.path.join(self.processed_data_path, filename)
        df.to_csv(output_path, index=False)
        print(f"💾 Processed data saved to {output_path}")
        
        # Save metadata
        metadata = {
            'processing_date': datetime.now().isoformat(),
            'total_samples': len(df),
            'num_features': len([col for col in df.columns if col.startswith('landmark_')]),
            'gesture_classes': df['gesture'].unique().tolist(),
            'statistics': self.stats,
            'config': self.config
        }
        
        metadata_path = os.path.join(self.processed_data_path, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"📄 Metadata saved to {metadata_path}")
    
    def create_train_test_split(self, df: pd.DataFrame, test_size: float = 0.2):
        """
        Create train/test split and save separately
        
        Args:
            df: Full dataset DataFrame
            test_size: Proportion of test data
        """
        from sklearn.model_selection import train_test_split
        
        if df.empty:
            print("⚠️ No data to split")
            return
        
        # Stratified split based on gesture
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=42, 
            stratify=df['gesture']
        )
        
        # Save train set
        train_path = os.path.join(self.processed_data_path, 'train_landmarks.csv')
        train_df.to_csv(train_path, index=False)
        print(f"💾 Training data saved to {train_path} ({len(train_df)} samples)")
        
        # Save test set
        test_path = os.path.join(self.processed_data_path, 'test_landmarks.csv')
        test_df.to_csv(test_path, index=False)
        print(f"💾 Test data saved to {test_path} ({len(test_df)} samples)")
        
        return train_df, test_df
    
    def validate_processed_data(self, df: pd.DataFrame) -> Dict:
        """
        Validate processed data quality
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Validation results
        """
        if df.empty:
            return {'valid': False, 'reason': 'Empty dataset'}
        
        validation_results = {
            'valid': True,
            'total_samples': len(df),
            'num_features': len([col for col in df.columns if col.startswith('landmark_')]),
            'gesture_classes': len(df['gesture'].unique()),
            'missing_values': df.isnull().sum().sum(),
            'class_distribution': df['gesture'].value_counts().to_dict()
        }
        
        # Check for minimum samples per class
        min_samples_per_class = 10
        low_sample_classes = []
        
        for gesture, count in validation_results['class_distribution'].items():
            if count < min_samples_per_class:
                low_sample_classes.append(gesture)
        
        if low_sample_classes:
            validation_results['warnings'] = f"Classes with < {min_samples_per_class} samples: {low_sample_classes}"
        
        # Check for missing landmarks
        expected_landmarks = 63  # 21 landmarks * 3 coordinates
        actual_landmarks = validation_results['num_features']
        
        if actual_landmarks != expected_landmarks:
            validation_results['valid'] = False
            validation_results['reason'] = f"Expected {expected_landmarks} landmark features, got {actual_landmarks}"
        
        print(f"📋 Validation Results:")
        print(f"  Valid: {validation_results['valid']}")
        print(f"  Total samples: {validation_results['total_samples']}")
        print(f"  Features: {validation_results['num_features']}")
        print(f"  Classes: {validation_results['gesture_classes']}")
        print(f"  Missing values: {validation_results['missing_values']}")
        
        return validation_results
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'hands'):
            self.hands.close()
        print("🧹 Preprocessor cleaned up")

def main():
    """Main preprocessing function"""
    print("🚀 Starting ISL Gesture Data Preprocessing")
    
    # Configuration
    config = {
        'raw_data_path': 'data/raw/gestures',
        'processed_data_path': 'data/processed',
        'normalize_landmarks': True,
        'augment_data': True
    }
    
    # Initialize preprocessor
    preprocessor = GestureDataPreprocessor(config)
    
    # Check if raw data exists
    if not os.path.exists(config['raw_data_path']):
        print(f"📝 No raw data found at {config['raw_data_path']}")
        print("🎲 Generating sample data structure...")
        
        # Create sample data structure
        sample_gestures = ['A', 'B', 'C', 'HELLO', 'THANK', '1', '2', '3']
        os.makedirs(config['raw_data_path'], exist_ok=True)
        
        for gesture in sample_gestures:
            gesture_dir = os.path.join(config['raw_data_path'], gesture)
            os.makedirs(gesture_dir, exist_ok=True)
            print(f"  Created directory: {gesture_dir}")
        
        print("📁 Sample directory structure created. Add your gesture images to these folders.")
        print("   Expected structure: data/raw/gestures/GESTURE_NAME/*.jpg")
        return
    
    # Process dataset
    df = preprocessor.process_dataset_folder(config['raw_data_path'])
    
    if df.empty:
        print("❌ No data was processed. Check your dataset structure and images.")
        return
    
    # Validate processed data
    validation_results = preprocessor.validate_processed_data(df)
    
    if not validation_results['valid']:
        print(f"❌ Data validation failed: {validation_results['reason']}")
        return
    
    # Save processed data
    preprocessor.save_processed_data(df)
    
    # Create train/test split
    train_df, test_df = preprocessor.create_train_test_split(df)
    
    # Cleanup
    preprocessor.cleanup()
    
    print("✅ Data preprocessing completed successfully!")
    print(f"📊 Final statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Training samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")
    print(f"  Gesture classes: {len(df['gesture'].unique())}")

if __name__ == "__main__":
    main()