#!/usr/bin/env python3
"""
Test script to verify gesture recognition functionality
"""
import cv2
import numpy as np
from config import config
from src.models.gesture_recognizer import GestureRecognizer

def test_gesture_recognition():
    """Test gesture recognition with camera"""
    print("🚀 Testing Gesture Recognition System")
    print("=" * 50)
    
    # Initialize configuration
    app_config = config['development']()
    
    model_config = {
        'MAX_NUM_HANDS': app_config.MAX_NUM_HANDS,
        'HAND_DETECTION_CONFIDENCE': app_config.HAND_DETECTION_CONFIDENCE,
        'HAND_TRACKING_CONFIDENCE': app_config.HAND_TRACKING_CONFIDENCE
    }
    
    print(f"📱 Camera index: {app_config.CAMERA_INDEX}")
    print(f"🎯 Detection confidence: {app_config.HAND_DETECTION_CONFIDENCE}")
    print(f"📍 Tracking confidence: {app_config.HAND_TRACKING_CONFIDENCE}")
    print(f"📊 Model path: {app_config.CNN_MODEL_PATH}")
    
    # Initialize gesture recognizer
    try:
        recognizer = GestureRecognizer(
            model_path=app_config.CNN_MODEL_PATH,
            config=model_config
        )
        print("✅ Gesture recognizer initialized")
    except Exception as e:
        print(f"❌ Error initializing recognizer: {e}")
        return
    
    # Initialize camera
    try:
        cap = cv2.VideoCapture(app_config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, app_config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, app_config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, app_config.CAMERA_FPS)
        
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
        
        print("✅ Camera initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing camera: {e}")
        return
    
    print("\n🎥 Starting camera test...")
    print("📝 Instructions:")
    print("   - Show your hand to the camera")
    print("   - Make different ISL gestures")
    print("   - Press 'q' to quit")
    print("   - Press 's' to show model info")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break
        
        # Flip frame horizontally
        frame = cv2.flip(frame, 1)
        frame_count += 1
        
        # Test gesture recognition
        try:
            result = recognizer.predict(frame)
            
            if result:
                # Draw landmarks
                landmarks_data = result.get('landmarks_data')
                frame = recognizer.draw_landmarks(frame, landmarks_data)
                
                # Display results
                gesture = result.get('gesture', 'Unknown')
                confidence = result.get('confidence', 0.0)
                hand_count = result.get('hand_count', 0)
                prediction_time = result.get('prediction_time', 0.0)
                
                # Add text overlay
                cv2.putText(frame, f"Hands: {hand_count}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(frame, f"Gesture: {gesture}", (10, 70),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence:.3f}", (10, 110),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Time: {prediction_time*1000:.1f}ms", (10, 150),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                
                # High confidence predictions
                if confidence > app_config.CONFIDENCE_THRESHOLD:
                    cv2.putText(frame, "DETECTED!", (10, 190),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
            else:
                cv2.putText(frame, "No hands detected", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
        except Exception as e:
            cv2.putText(frame, f"Error: {str(e)[:50]}", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            print(f"Frame {frame_count}: Error in recognition: {e}")
        
        # Show frame info
        cv2.putText(frame, f"Frame: {frame_count}", (frame.shape[1] - 150, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
        
        # Display frame
        cv2.imshow('ISL Gesture Recognition Test', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Show model info
            model_info = recognizer.get_model_info()
            print("\n📊 Model Information:")
            for key, value in model_info.items():
                if key != 'class_labels':  # Skip long class list
                    print(f"   {key}: {value}")
            if 'class_labels' in model_info:
                print(f"   class_labels: {len(model_info['class_labels'])} classes")
            print()
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    recognizer.cleanup()
    
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    test_gesture_recognition()