# 🤟 Indian Sign Language (ISL) to Text Conversion System

An AI-powered real-time gesture recognition system that interprets Indian Sign Language gestures and converts them into text and speech output. This system bridges the communication gap between the deaf/hearing-impaired community and non-sign language users.

## 🚀 Features

- **Real-time Gesture Recognition**: Live ISL gesture detection using webcam with MediaPipe hand tracking
- **Dual-Hand Support**: Recognizes both one-hand and two-hand gestures simultaneously
- **Comprehensive Alphabet & Numbers**: Supports ISL alphabets (A-Z) and digits (0-9)
- **Advanced Text-to-Speech**: Multiple TTS engines (pyttsx3, gTTS) with customizable voice settings
- **Intelligent NLP Processing**: Context-aware sentence formation with word completion and suggestions
- **Dynamic Gestures**: Support for motion-based gestures using LSTM networks (planned)
- **Interactive Web Interface**: Modern dashboard with live video feed and real-time feedback
- **Educational Games**: Alphabet and numbers learning games with progress tracking
- **Performance Monitoring**: Real-time accuracy, FPS, and system resource tracking
- **RESTful API**: Complete API endpoints for integration and custom applications
- **Multi-Platform Support**: Cross-platform compatibility (Windows, Linux, macOS)
- **Configurable System**: Extensive configuration options for different environments

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Browser   │    │   Flask Server   │    │   AI Models     │
│                 │    │                  │    │                 │
│ • HTML/CSS/JS   │◄──►│ • REST API       │◄──►│ • CNN Model     │
│ • Video Stream  │    │ • WebSocket      │    │ • LSTM Model    │
│ • Games         │    │ • Template Engine│    │ • Hand Tracking │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ MediaPipe Hand  │    │  NLP Processor   │    │ Text-to-Speech  │
│ Tracking        │    │                  │    │                 │
│ • Landmark      │◄──►│ • Word Formation │◄──►│ • pyttsx3       │
│   Detection     │    │ • Sentence Build │    │ • gTTS          │
│ • Pose Analysis │    │ • Auto-complete  │    │ • Voice Config  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
isl-gesture-recognition/
├── app.py                          # Main Flask application
├── run.py                          # Application launcher script
├── config.py                       # Configuration management
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── quick_test.py                   # Quick testing script
├── test_gesture_recognition.py     # Unit tests
│
├── src/                            # Source code
│   ├── __init__.py
│   ├── models/                     # Machine learning models
│   │   ├── __init__.py
│   │   ├── gesture_recognizer.py   # Main recognition class
│   │   └── train_cnn.py           # CNN model training
│   ├── utils/                      # Utility modules
│   │   ├── __init__.py
│   │   ├── tts_handler.py         # Text-to-speech handler
│   │   ├── nlp_processor.py       # Natural language processing
│   │   └── performance_monitor.py # Performance monitoring
│   └── preprocessing/              # Data preprocessing
│       ├── __init__.py
│       └── preprocess_data.py     # Data preparation scripts
│
├── static/                         # Static web assets
│   ├── css/
│   │   ├── style.css              # Main stylesheet
│   │   └── style-game.css        # Game-specific styles
│   ├── js/
│   │   └── app.js                # Frontend JavaScript
│   ├── images/                   # Static images
│   ├── gestures/                 # Gesture demonstration images
│   └── uploads/                  # User upload directory
│
├── templates/                     # HTML templates
│   ├── index.html                # Main dashboard
│   ├── learn.html                # Learning interface
│   ├── alphabet_game.html        # Alphabet learning game
│   ├── numbers_game.html         # Numbers learning game
│   └── errors/                   # Error page templates
│       ├── 404.html
│       └── 500.html
│
├── data/                         # Data management
│   ├── raw/                      # Raw gesture data
│   ├── processed/                # Preprocessed datasets
│   └── datasets/                 # Training datasets
│
├── models/                       # Trained models and artifacts
│   ├── trained/                  # Production models
│   │   ├── cnn_model.h5         # Trained CNN model
│   │   ├── cnn_model_config.json # Model configuration
│   │   └── *_labels.pkl         # Class labels
│   ├── checkpoints/              # Training checkpoints
│   ├── plots/                    # Training visualizations
│   └── evaluation_results.json   # Model evaluation metrics
│
├── logs/                         # Application logs
│   └── performance.log           # Performance monitoring logs
│
├── tests/                        # Test suites
├── docs/                         # Documentation
└── frontend/                     # Alternative frontend (legacy)
    ├── index.html
    ├── learn.html
    ├── alphabet-game.html
    └── static/
```

## 🔧 Installation

### Prerequisites

- **Python 3.7+**: Required for TensorFlow compatibility
- **Webcam**: For real-time gesture recognition
- **Git**: For cloning the repository

## 🎯 Usage

### Basic Usage

1. **Start the Application**: Run `python run.py`
2. **Allow Camera Access**: Grant webcam permissions when prompted
3. **Position Hands**: Place hands clearly in front of camera (6-12 inches away)
4. **Make Gestures**: Perform ISL signs for letters A-Z or numbers 0-9
5. **View Results**: See real-time text conversion on screen
6. **Listen to Speech**: Click "Speak" button to hear the text

### Advanced Features

#### Real-time Gesture Recognition
- **Confidence Threshold**: Gestures below 60% confidence are ignored
- **Gesture Interval**: New gestures captured every 2 seconds
- **Multi-hand Support**: Recognizes gestures from both hands simultaneously

#### Text-to-Speech
- **Multiple Engines**: Automatic fallback between pyttsx3 and gTTS
- **Voice Customization**: Adjustable rate, volume, and voice selection
- **Async Processing**: Non-blocking speech synthesis

#### NLP Processing
- **Word Completion**: Automatic completion of partial words
- **Sentence Formation**: Intelligent sentence building from gesture sequences
- **Word Suggestions**: Context-aware word suggestions

#### Educational Games
- **Alphabet Game**: Interactive alphabet learning with scoring
- **Numbers Game**: Number recognition practice
- **Progress Tracking**: Learning progress visualization

### Keyboard Shortcuts

- **Space**: Speak current sentence
- **Enter**: Add gesture to sentence
- **Backspace**: Clear last gesture
- **Ctrl+C**: Clear entire sentence

## 📡 API Documentation

### Core Endpoints

#### GET `/`
Main dashboard interface

#### GET `/video_feed`
MJPEG video stream with gesture recognition overlay

#### GET `/api/gesture_data`
Get current gesture recognition data
```json
{
  "current_sentence": "HELLO WORLD",
  "current_gesture": "H",
  "confidence": 0.95,
  "hand_count": 1,
  "fps": 28.5,
  "timestamp": 1640995200.0
}
```

#### POST `/api/speak`
Convert text to speech
```json
{
  "text": "Hello World",
  "success": true
}
```

#### POST `/api/translate`
Translate text to Hindi
```json
{
  "text": "Hello World",
  "translated": "नमस्ते दुनिया"
}
```

#### POST `/api/update_sentence`
Update current sentence
```json
{
  "sentence": "New sentence text"
}
```

#### POST `/api/clear_sentence`
Clear current sentence

#### GET `/api/performance`
Get performance metrics
```json
{
  "fps": 28.5,
  "avg_fps": 27.8,
  "accuracy": 0.94,
  "latency": 45.2,
  "system": {
    "cpu_percent": 45.2,
    "memory_percent": 62.1
  }
}
```

#### GET `/api/model_info`
Get model information
```json
{
  "model_type": "CNN",
  "input_shape": [null, 63],
  "num_classes": 37,
  "class_labels": ["A", "B", "C", ...],
  "avg_prediction_time": 0.035
}
```

### Game Endpoints

#### GET `/learn`
Learning interface

#### GET `/alphabet-game`
Alphabet learning game

#### GET `/numbers-game`
Numbers learning game

### Health Check

#### GET `/health`
System health status
```json
{
  "status": "healthy",
  "gesture_recognizer": true,
  "camera_available": true,
  "tts_available": true,
  "timestamp": 1640995200.0
}
```

#### GET `/debug`
Detailed debug information

## 🧠 Model Training

### Training Requirements

- **Dataset**: ISL gesture images (A-Z, 0-9)
- **Hardware**: GPU recommended for faster training
- **Time**: 2-4 hours for full training

### Training Process

1. **Prepare Dataset**
   ```bash
   python src/preprocessing/preprocess_data.py
   ```

2. **Train CNN Model**
   ```python
   from src.models.train_cnn import train_cnn_model

   # Train the model
   model = train_cnn_model(
       data_dir='data/processed',
       epochs=50,
       batch_size=32,
       validation_split=0.2
   )
   ```

3. **Evaluate Model**
   ```python
   from src.models.gesture_recognizer import GestureRecognizer

   # Load and evaluate
   recognizer = GestureRecognizer()
   metrics = recognizer.evaluate_model()
   ```

4. **Save Model**
   ```python
   recognizer.save_model('models/trained/cnn_model.h5')
   ```

### Model Configuration

```python
# Model hyperparameters
config = {
    'input_shape': (63,),  # 21 landmarks * 3 coordinates
    'num_classes': 37,     # A-Z + 0-9 + special symbols
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'dropout_rate': 0.3
}
```

### Configuration Classes

- **DevelopmentConfig**: Debug mode, detailed logging
- **ProductionConfig**: Optimized for production deployment
- **TestingConfig**: Disabled camera, mock data

### Performance Optimization

1. **Reduce Camera Resolution**
2. **Lower Confidence Threshold**
3. **Disable Performance Logging**
4. **Use GPU for Inference**
5. **Optimize Model Size**

## 📊 Performance Metrics

### Current Benchmarks

- **Accuracy**: ~95% on test dataset (A-Z, 0-9)
- **FPS**: 25-30 frames per second
- **Latency**: 35-50ms per prediction
- **Memory Usage**: ~500MB RAM
- **CPU Usage**: 40-60% during recognition

## 🛠️ Technical Stack

### Backend
- **Framework**: Flask 2.3.3
- **Language**: Python 3.7+
- **ASGI Server**: Werkzeug (development), Gunicorn (production)

### Computer Vision & AI
- **Computer Vision**: OpenCV 4.8.1
- **Hand Tracking**: MediaPipe 0.10.7
- **Deep Learning**: TensorFlow 2.13.0, Keras 2.13.1
- **Data Processing**: NumPy 1.24.3, Pandas 2.0.3

### Natural Language Processing
- **Core NLP**: NLTK 3.8.1, spaCy 3.6.1
- **Text Processing**: scikit-learn 1.3.0

### Text-to-Speech
- **Primary Engine**: pyttsx3 2.90
- **Fallback Engine**: gTTS 2.3.2
- **Audio Processing**: pygame (for gTTS)

### Frontend
- **HTML5**: Semantic markup
- **CSS3**: Responsive design, animations
- **JavaScript**: ES6+ features, async/await
- **Real-time Updates**: Server-Sent Events

### Development & Testing
- **Testing**: pytest 7.4.2, pytest-flask 1.2.0
- **Code Quality**: black 23.7.0, flake8 6.0.0
- **Type Checking**: mypy (optional)
- **Performance**: psutil 5.9.5

### Deployment
- **Containerization**: Docker, docker-compose
- **Process Management**: systemd, supervisor
- **Reverse Proxy**: Nginx
- **SSL/TLS**: Let's Encrypt

## 🔮 Future Enhancements

### Short Term (3-6 months)
- [ ] Expand vocabulary to 500+ common ISL words
- [ ] Add dynamic gesture recognition (motion-based)
- [ ] Implement user authentication and profiles
- [ ] Add gesture recording and custom model training
- [ ] Mobile application (React Native)

### Medium Term (6-12 months)
- [ ] Multi-language support (Hindi, regional languages)
- [ ] Real-time collaboration features
- [ ] Cloud deployment with auto-scaling
- [ ] Integration with video conferencing platforms
- [ ] Advanced NLP with context understanding

### Long Term (1-2 years)
- [ ] Full ISL grammar and syntax support
- [ ] Emotion recognition from facial expressions
- [ ] Sign language translation between different regions
- [ ] Educational platform with curriculum
- [ ] Professional interpreter assistance system

### Technical Improvements
- [ ] Model optimization (quantization, pruning)
- [ ] Edge computing deployment
- [ ] Real-time model updates
- [ ] Multi-modal input (voice + gesture)
- [ ] Advanced computer vision techniques  

### Contribution Guidelines

- **Code Style**: Follow PEP 8 and use Black formatter
- **Testing**: Maintain >80% test coverage
- **Documentation**: Update README and docstrings
- **Commits**: Use conventional commit messages
- **Issues**: Use issue templates for bug reports and features

### Areas for Contribution

- **Model Improvement**: Better accuracy, new gesture types
- **UI/UX Enhancement**: Better user interface and experience
- **Performance Optimization**: Faster inference, lower latency
- **Documentation**: Tutorials, API docs, video guides
- **Testing**: More comprehensive test suites
- **Internationalization**: Support for other sign languages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Permissions
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use

### Limitations
- ❌ Liability
- ❌ Warranty

### Conditions
- 📝 License and copyright notice

## 🙏 Acknowledgments

### Core Technologies
- **MediaPipe Team**: Revolutionary hand tracking technology
- **TensorFlow Team**: Powerful deep learning framework
- **OpenCV Community**: Computer vision excellence
- **Flask Team**: Lightweight web framework

### Research & Data
- **ISL Research Community**: Gesture datasets and research papers
- **Deaf Community**: Valuable feedback and testing
- **Educational Institutions**: Research collaborations

---

**Made with ❤️ for the deaf and hearing-impaired community**
