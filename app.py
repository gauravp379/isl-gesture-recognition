"""
Main Flask application for ISL Gesture Recognition System
"""
import os
import cv2
import json
import time
from flask import Flask, render_template, jsonify, request, Response
from flask_cors import CORS
import numpy as np
import mediapipe as mp
from threading import Thread
import queue

# Import custom modules
from config import config
from src.models.gesture_recognizer import GestureRecognizer
from src.utils.tts_handler import TTSHandler
from src.utils.nlp_processor import NLPProcessor
from src.utils.performance_monitor import PerformanceMonitor

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(config['development'])
CORS(app)

# Initialize components
gesture_recognizer = None
tts_handler = TTSHandler()
nlp_processor = NLPProcessor()
performance_monitor = PerformanceMonitor()

# Global variables for video streaming
camera = None
output_frame = None
lock = Thread()
gesture_queue = queue.Queue()
current_sentence = ""

def initialize_components():
    """Initialize all system components"""
    global gesture_recognizer
    try:
        gesture_recognizer = GestureRecognizer()
        print("✅ Gesture recognizer initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing gesture recognizer: {e}")
        gesture_recognizer = None

def get_camera():
    """Initialize and return camera object"""
    global camera
    if camera is None:
        camera = cv2.VideoCapture(app.config['CAMERA_INDEX'])
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, app.config['CAMERA_WIDTH'])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, app.config['CAMERA_HEIGHT'])
        camera.set(cv2.CAP_PROP_FPS, app.config['CAMERA_FPS'])
    return camera

def generate_frames():
    """Generate video frames with gesture recognition"""
    global output_frame, current_sentence
    
    camera = get_camera()
    frame_count = 0
    start_time = time.time()
    
    while True:
        success, frame = camera.read()
        if not success:
            break
            
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Perform gesture recognition
        try:
            if gesture_recognizer:
                result = gesture_recognizer.predict(frame)
                
                if result:
                    # Draw hand landmarks and predictions
                    frame = gesture_recognizer.draw_landmarks(frame)
                    
                    # Display prediction
                    gesture = result.get('gesture', '')
                    confidence = result.get('confidence', 0.0)
                    
                    if confidence > app.config['CONFIDENCE_THRESHOLD']:
                        # Add recognized gesture to queue
                        gesture_queue.put(gesture)
                        
                        # Update current sentence
                        current_sentence = process_gesture_sequence()
                        
                        # Display on frame
                        cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"Sentence: {current_sentence}", (10, 110),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
        except Exception as e:
            print(f"Error in gesture recognition: {e}")
        
        # Calculate and display FPS
        frame_count += 1
        if frame_count % 30 == 0:
            end_time = time.time()
            fps = 30 / (end_time - start_time)
            performance_monitor.log_fps(fps)
            start_time = end_time
            
        cv2.putText(frame, f"FPS: {fps:.1f}" if 'fps' in locals() else "FPS: --", 
                   (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Store frame globally
        output_frame = frame
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def process_gesture_sequence():
    """Process sequence of gestures to form meaningful text"""
    gestures = []
    while not gesture_queue.empty():
        gestures.append(gesture_queue.get())
    
    if gestures:
        # Use NLP processor to form sentences
        return nlp_processor.process_gestures(gestures[-10:])  # Last 10 gestures
    return current_sentence

@app.route('/')
def index():
    """Main dashboard route"""
    return render_template('index.html', 
                         gesture_classes=app.config['GESTURE_CLASSES'])

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/gesture_data')
def get_gesture_data():
    """API endpoint to get current gesture recognition data"""
    try:
        data = {
            'current_sentence': current_sentence,
            'fps': performance_monitor.get_current_fps(),
            'confidence': 0.0,
            'timestamp': time.time()
        }
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/speak', methods=['POST'])
def speak_text():
    """API endpoint for text-to-speech conversion"""
    try:
        data = request.get_json()
        text = data.get('text', current_sentence)
        
        if text:
            success = tts_handler.speak(text)
            return jsonify({'success': success, 'text': text})
        else:
            return jsonify({'error': 'No text to speak'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear_sentence', methods=['POST'])
def clear_sentence():
    """API endpoint to clear current sentence"""
    global current_sentence
    current_sentence = ""
    
    # Clear gesture queue
    while not gesture_queue.empty():
        gesture_queue.get()
    
    return jsonify({'success': True, 'message': 'Sentence cleared'})

@app.route('/api/performance')
def get_performance_data():
    """API endpoint to get performance metrics"""
    try:
        metrics = performance_monitor.get_metrics()
        return jsonify(metrics)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/gesture_classes')
def get_gesture_classes():
    """API endpoint to get available gesture classes"""
    return jsonify({
        'classes': app.config['GESTURE_CLASSES'],
        'num_classes': app.config['NUM_CLASSES']
    })

@app.route('/api/model_info')
def get_model_info():
    """API endpoint to get model information"""
    try:
        if gesture_recognizer:
            model_info = gesture_recognizer.get_model_info()
            return jsonify(model_info)
        else:
            return jsonify({'error': 'Model not loaded'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'gesture_recognizer': gesture_recognizer is not None,
        'camera_available': camera is not None,
        'tts_available': tts_handler.is_available(),
        'timestamp': time.time()
    }
    return jsonify(status)

@app.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('errors/500.html'), 500

if __name__ == '__main__':
    print("🚀 Initializing ISL Gesture Recognition System...")
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Initialize components
    initialize_components()
    
    print(f"📹 Camera index: {app.config['CAMERA_INDEX']}")
    print(f"🎯 Confidence threshold: {app.config['CONFIDENCE_THRESHOLD']}")
    print(f"📊 Number of gesture classes: {app.config['NUM_CLASSES']}")
    
    # Start the Flask application
    print("🌐 Starting web server...")
    print("📱 Open your browser and go to: http://localhost:5000")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=app.config['DEBUG'],
        threaded=True
    )