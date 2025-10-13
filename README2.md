# 🤟 Indian Sign Language (ISL) to Text Conversion System

An AI-powered real-time gesture recognition system that interprets Indian Sign Language gestures and converts them into text and speech output. This system bridges the communication gap between the deaf/hearing-impaired community and non-sign language users.

## 🚀 Features

- **Real-time Gesture Recognition**: Live ISL gesture detection using webcam
- **Dual-Hand Support**: Recognizes both one-hand and two-hand gestures
- **Alphabet & Numbers**: Supports ISL alphabets (A-Z) and digits (0-9)
- **Text-to-Speech**: Converts recognized text to audible speech
- **Dynamic Gestures**: Handles motion-based gestures using LSTM networks
- **Web Interface**: Interactive dashboard with live video feed
- **Performance Monitoring**: Real-time accuracy and FPS tracking
- **NLP Integration**: Context-aware sentence formation

## 🏗️ Architecture

```
User Gesture → Camera Capture → MediaPipe Hand Tracking → 
Feature Extraction → CNN/LSTM Model → NLP Processing → 
Text Output → Speech Synthesis
```

## 📁 Project Structure

```
isl-gesture-recognition/
├── src/
│   ├── models/           # Deep learning models (CNN, LSTM)
│   ├── utils/            # Utility functions
│   └── preprocessing/    # Data preprocessing modules
├── static/
│   ├── css/             # Stylesheets
│   ├── js/              # JavaScript files
│   └── images/          # Static images
├── templates/           # HTML templates
├── data/
│   ├── raw/            # Raw gesture data
│   ├── processed/      # Preprocessed data
│   └── datasets/       # Training datasets
├── models/
│   ├── trained/        # Trained model files
│   └── checkpoints/    # Model checkpoints
├── tests/              # Unit tests
├── docs/               # Documentation
├── app.py              # Main Flask application
├── requirements.txt    # Dependencies
└── README.md          # Project documentation
```

## 🔧 Installation

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
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open browser and navigate to**
   ```
   http://localhost:5000
   ```

## 🎯 Usage

1. **Start the Application**: Run `python app.py`
2. **Allow Camera Access**: Grant webcam permissions when prompted
3. **Position Hands**: Place hands clearly in front of camera
4. **Make Gestures**: Perform ISL signs for letters A-Z or numbers 0-9
5. **View Results**: See real-time text conversion on screen
6. **Listen to Speech**: Click "Speak" to hear the text

## 🧠 Model Architecture

### Static Gesture Recognition
- **CNN Model**: Convolutional layers for feature extraction
- **Input**: Hand landmark coordinates (21 points per hand)
- **Output**: Probability distribution over 36 classes (A-Z, 0-9)

### Dynamic Gesture Recognition
- **CNN + LSTM**: Temporal sequence modeling
- **Input**: Sequential frames of hand movements
- **Output**: Motion-based gesture classification

## 📊 Performance Metrics

- **Accuracy**: ~95% on test dataset
- **FPS**: 30+ frames per second
- **Latency**: <100ms gesture-to-text conversion
- **Supported Gestures**: 36 (26 letters + 10 digits)

## 🛠️ Technical Stack

- **Backend**: Flask (Python)
- **Computer Vision**: OpenCV, MediaPipe
- **Deep Learning**: TensorFlow/Keras
- **Frontend**: HTML5, CSS3, JavaScript
- **Text-to-Speech**: pyttsx3, gTTS
- **NLP**: NLTK, spaCy

## 🔮 Future Enhancements

- [ ] Expand to full ISL vocabulary (words & sentences)
- [ ] Multi-language support (Hindi, regional languages)
- [ ] Mobile app development
- [ ] Cloud deployment with API endpoints
- [ ] Real-time collaboration features
- [ ] Educational module for ISL learning

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MediaPipe team for hand tracking technology
- TensorFlow team for deep learning framework
- ISL research community for gesture datasets
- Open source contributors

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please reach out through the project issues or contact the development team.

---

**Made with ❤️ for the deaf and hearing-impaired community**
