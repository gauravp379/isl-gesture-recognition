"""
Configuration file for ISL Gesture Recognition System
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Base configuration class"""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'isl-gesture-recognition-secret-key'
    DEBUG = False
    TESTING = False
    
    # Camera Configuration
    CAMERA_INDEX = 0
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    
    # Model Configuration
    MODEL_PATH = os.path.join('models', 'trained')
    CNN_MODEL_PATH = os.path.join(MODEL_PATH, 'cnn_model.h5')
    LSTM_MODEL_PATH = os.path.join(MODEL_PATH, 'lstm_model.h5')
    
    # Gesture Recognition Settings
    CONFIDENCE_THRESHOLD = 0.6
    HAND_DETECTION_CONFIDENCE = 0.5  # Lowered for better detection
    HAND_TRACKING_CONFIDENCE = 0.3   # Lowered for better tracking
    MAX_NUM_HANDS = 2
    
    # Data Paths
    DATA_DIR = 'data'
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    DATASETS_DIR = os.path.join(DATA_DIR, 'datasets')
    
    # Static Files
    STATIC_DIR = 'static'
    UPLOAD_FOLDER = os.path.join(STATIC_DIR, 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi'}
    
    # Text-to-Speech Configuration
    TTS_ENGINE = 'pyttsx3'  # or 'gtts'
    TTS_RATE = 150
    TTS_VOLUME = 0.8
    TTS_VOICE_INDEX = 0  # 0 for male, 1 for female
    
    # NLP Configuration
    NLTK_DATA_PATH = os.path.join(os.getcwd(), 'nltk_data')
    SPACY_MODEL = 'en_core_web_sm'
    
    # Performance Monitoring
    LOG_PERFORMANCE = True
    LOG_FILE = 'logs/performance.log'
    FPS_LOG_INTERVAL = 30  # Log FPS every 30 seconds
    
    # Gesture Classes
    GESTURE_CLASSES = [
        # Alphabets
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z',
        # Numbers
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        # Special symbols
        '+'
    ]
    
    NUM_CLASSES = len(GESTURE_CLASSES)

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    CAMERA_FPS = 30
    LOG_PERFORMANCE = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    CAMERA_FPS = 60
    CONFIDENCE_THRESHOLD = 0.8
    LOG_PERFORMANCE = True

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    CAMERA_INDEX = -1  # Disable camera for testing
    DEBUG = True

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}