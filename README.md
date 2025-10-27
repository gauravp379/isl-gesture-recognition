# 🤟 Indian Sign Language (ISL) Gesture Recognition System

An AI-powered real-time gesture recognition system that converts Indian Sign Language gestures into text and speech, bridging communication gaps for the deaf/hearing-impaired community.

## 🚀 Key Features

- **Real-time Recognition**: Live ISL gesture detection using webcam and MediaPipe hand tracking
- **Comprehensive Support**: Recognizes alphabets (A-Z) and numbers (0-9) with dual-hand support
- **Text-to-Speech**: Multiple TTS engines (pyttsx3, gTTS) for audio output
- **NLP Processing**: Context-aware sentence formation and word suggestions
- **Web Interface**: Modern dashboard with live video feed and real-time feedback
- **Educational Games**: Interactive alphabet and numbers learning games
- **RESTful API**: Endpoints for integration and custom applications

## 🏗️ Project Structure

```
isl-gesture-recognition/
├── app.py                 # Main Flask application
├── run.py                 # Application launcher
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── src/                   # Source code (models, utils, preprocessing)
├── static/                # Web assets (CSS, JS, images)
├── templates/             # HTML templates
├── data/                  # Datasets and processed data
├── models/                # Trained models and checkpoints
├── logs/                  # Performance logs
└── tests/                 # Test suites
```

## 🔧 Installation

### Prerequisites
- Python 3.7+
- Webcam
- Git

### Step-by-Step Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd isl-gesture-recognition
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

5. **Run the application**
   ```bash
   python run.py
   ```

6. **Open browser**
   Navigate to `http://localhost:5000`

## 🎯 Usage

1. **Start the App**: Run `python run.py` and open `http://localhost:5000`
2. **Grant Camera Access**: Allow webcam permissions
3. **Position Hands**: Place hands 6-12 inches from camera
4. **Make Gestures**: Perform ISL signs for letters/numbers
5. **View Results**: See real-time text conversion
6. **Hear Speech**: Click "Speak" for audio output

### Keyboard Shortcuts
- **Space**: Speak sentence
- **Enter**: Add gesture to sentence
- **Backspace**: Clear last gesture
- **Ctrl+C**: Clear sentence

## 📡 API Endpoints

- `GET /`: Main dashboard
- `GET /video_feed`: MJPEG video stream
- `GET /api/gesture_data`: Current gesture data
- `POST /api/speak`: Text-to-speech
- `GET /health`: System health check

## 🧠 Model Training

Train custom models using `src/models/train_cnn.py` with prepared datasets in `data/`.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**Made with ❤️ for accessibility and inclusion**
