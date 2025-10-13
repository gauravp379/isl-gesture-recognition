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

# Initialize gesture recognizer immediately
print("🚀 Initializing ISL Gesture Recognition System...")
try:
    # Pass model path and configuration to gesture recognizer
    model_config = {
        'MAX_NUM_HANDS': config['development'].MAX_NUM_HANDS,
        'HAND_DETECTION_CONFIDENCE': config['development'].HAND_DETECTION_CONFIDENCE,
        'HAND_TRACKING_CONFIDENCE': config['development'].HAND_TRACKING_CONFIDENCE
    }
    
    gesture_recognizer = GestureRecognizer(
        model_path=config['development'].CNN_MODEL_PATH,
        config=model_config
    )
    print("✅ Gesture recognizer initialized successfully")
except Exception as e:
    print(f"❌ Error initializing gesture recognizer: {e}")
    gesture_recognizer = None

# Global variables for video streaming
camera = None
output_frame = None
lock = Thread()
gesture_queue = queue.Queue()
current_sentence = ""

# Global variables for real-time gesture data
current_gesture = ""
current_confidence = 0.0
current_hand_count = 0
last_detection_time = 0
last_gesture_time = 0
GESTURE_CAPTURE_INTERVAL = 2.0  # Capture gesture every 2 seconds


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
    global output_frame, current_sentence, current_gesture, current_confidence, current_hand_count, last_gesture_time
    
    print(f"📹 Starting video stream generation...")
    print(f"🤖 Gesture recognizer status: {'Loaded' if gesture_recognizer else 'Not loaded'}")
    print(f"⏰ Gesture capture interval: {GESTURE_CAPTURE_INTERVAL} seconds")
    
    camera = get_camera()
    if not camera.isOpened():
        print("❌ Camera failed to open")
        return
    
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    while True:
        success, frame = camera.read()
        if not success:
            print("❌ Failed to read camera frame")
            break
            
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Perform gesture recognition
        try:
            current_time = time.time()
            if gesture_recognizer:
                result = gesture_recognizer.predict(frame)
                
                if result:
                    # Draw hand landmarks and predictions
                    landmarks_data = result.get('landmarks_data')
                    frame = gesture_recognizer.draw_landmarks(frame, landmarks_data)
                    
                    # Get prediction data
                    gesture = result.get('gesture', '')
                    confidence = result.get('confidence', 0.0)
                    hand_count = result.get('hand_count', 0)
                    
                    # Update real-time global variables (for display purposes)
                    current_gesture = gesture
                    current_confidence = confidence
                    current_hand_count = hand_count
                    
                    # Always show detection info
                    cv2.putText(frame, f"Hands: {hand_count}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(frame, f"Gesture: {gesture}", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 90),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Check if it's time to capture a gesture (every 2 seconds)
                    if (current_time - last_gesture_time) >= GESTURE_CAPTURE_INTERVAL:
                        # Only capture if hands are detected and confidence is high enough
                        if hand_count > 0 and confidence > app.config['CONFIDENCE_THRESHOLD']:
                            # Add recognized gesture to queue
                            gesture_queue.put(gesture)

                            # Update current sentence (only add to sentence, don't output yet)
                            global current_sentence
                            current_sentence = nlp_processor.process_gesture_sequence([gesture], current_sentence)

                            # Don't speak immediately - wait for capture interval to complete

                            # Update last gesture time
                            last_gesture_time = current_time

                            print(f"✅ Captured gesture '{gesture}' with confidence {confidence:.2f}")
                    
                    # Display sentence
                    cv2.putText(frame, f"Sentence: {current_sentence}", (10, 120),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    # Show next capture countdown
                    next_capture_time = GESTURE_CAPTURE_INTERVAL - (current_time - last_gesture_time)
                    if next_capture_time > 0:
                        cv2.putText(frame, f"Next capture: {next_capture_time:.1f}s", (10, 150),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    # Show "No hands detected" when no result
                    cv2.putText(frame, "No hands detected", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Reset real-time variables when no hands detected
                    current_gesture = ""
                    current_confidence = 0.0
                    current_hand_count = 0
                
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
        # Process gestures using NLP processor with intelligent sentence building
        # Only process the last unique gesture to avoid repetition
        if len(gestures) > 0:
            # Take only the most recent gesture
            latest_gesture = gestures[-1]
            global current_sentence
            current_sentence = nlp_processor.process_gesture_sequence([latest_gesture], current_sentence)
            return current_sentence
    
    return current_sentence

@app.route('/')
def index():
    """Main dashboard route"""
    return render_template('index.html',
                         gesture_classes=app.config['GESTURE_CLASSES'])

@app.route('/learn')
def learn():
    """Learn mode route"""
    return render_template('learn.html')

@app.route('/alphabet-game')
def alphabet_game():
    """Alphabet game route"""
    return render_template('alphabet_game.html')

@app.route('/numbers-game')
def numbers_game():
    """Numbers game route"""
    return render_template('numbers_game.html')

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
            'current_gesture': current_gesture,
            'confidence': current_confidence,
            'hand_count': current_hand_count,
            'fps': performance_monitor.get_current_fps(),
            'last_detection_time': last_detection_time,
            'timestamp': time.time()
        }
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/suggestions', methods=['POST'])
def get_suggestions():
    """API endpoint for word suggestions"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        suggestions = nlp_processor.get_suggestions(text)
        
        return jsonify({
            'suggestions': suggestions,
            'text': text
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/update_sentence', methods=['POST'])
def update_sentence():
    """API endpoint to update current sentence from frontend"""
    try:
        data = request.get_json()
        global current_sentence
        current_sentence = data.get('sentence', '')
        
        # Clear gesture queue and buffers
        while not gesture_queue.empty():
            gesture_queue.get()
        nlp_processor.clear_buffers()
        
        return jsonify({'success': True, 'sentence': current_sentence})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/speak', methods=['POST'])
def speak_text():
    """API endpoint for text-to-speech conversion"""
    try:
        data = request.get_json()
        text = data.get('text', current_sentence)

        if text and text.strip():
            print(f"🗣️ Speaking text: '{text}'")
            print(f"🔊 TTS Handler available: {tts_handler.is_available()}")
            print(f"🔊 TTS Engine type: {tts_handler.engine_type}")
            success = tts_handler.speak(text.strip())
            print(f"✅ TTS result: {success}")
            return jsonify({'success': success, 'text': text})
        else:
            print("⚠️ No text provided for TTS")
            return jsonify({'error': 'No text to speak'}), 400

    except Exception as e:
        print(f"❌ TTS Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/translate', methods=['POST'])
def translate_text():
    """API endpoint for text translation to Hindi"""
    try:
        data = request.get_json()
        text = data.get('text', current_sentence)

        if text and text.strip():
            # Simple translation using Google Translate API (you might need to install googletrans)
            try:
                from googletrans import Translator
                translator = Translator()
                translation = translator.translate(text.strip(), src='en', dest='hi')
                hindi_text = translation.text
                print(f"🔄 Translated '{text}' to Hindi: '{hindi_text}'")
                return jsonify({'success': True, 'original': text, 'translated': hindi_text})
            except ImportError:
                # Fallback: return original text with note
                print("⚠️ googletrans not installed, returning original text")
                return jsonify({'success': False, 'error': 'Translation service not available', 'original': text})
            except Exception as trans_error:
                print(f"⚠️ Translation error: {trans_error}")
                return jsonify({'success': False, 'error': 'Translation failed', 'original': text})
        else:
            return jsonify({'error': 'No text to translate'}), 400

    except Exception as e:
        print(f"❌ Translation Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear_sentence', methods=['POST'])
def clear_sentence():
    """API endpoint to clear current sentence"""
    global current_sentence
    current_sentence = ""
    
    # Clear gesture queue
    while not gesture_queue.empty():
        gesture_queue.get()
    
    # Clear NLP processor buffers
    nlp_processor.clear_buffers()
    
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

@app.route('/debug')
def debug_info():
    """Debug information endpoint"""
    debug_data = {
        'gesture_recognizer_loaded': gesture_recognizer is not None,
        'camera_initialized': camera is not None,
        'model_path': app.config['CNN_MODEL_PATH'],
        'confidence_threshold': app.config['CONFIDENCE_THRESHOLD'],
        'hand_detection_confidence': app.config['HAND_DETECTION_CONFIDENCE'],
        'hand_tracking_confidence': app.config['HAND_TRACKING_CONFIDENCE'],
        'camera_index': app.config['CAMERA_INDEX']
    }
    
    if gesture_recognizer:
        try:
            model_info = gesture_recognizer.get_model_info()
            debug_data['model_info'] = model_info
        except Exception as e:
            debug_data['model_error'] = str(e)
    
    return jsonify(debug_data)

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
