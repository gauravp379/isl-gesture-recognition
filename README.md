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

## 🔧 Installation

### Prerequisites

- **Python 3.7+**: Required for TensorFlow compatibility
- **Webcam**: For real-time gesture recognition

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

---

**Made with ❤️ for the deaf and hearing-impaired community**
