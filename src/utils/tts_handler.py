"""
Text-to-Speech Handler for ISL Gesture Recognition System
Supports multiple TTS engines including pyttsx3 and gTTS
"""
import pyttsx3
import threading
import time
from typing import Optional, Dict, List
import os
import tempfile
import pygame
from gtts import gTTS
import io
import requests
import queue

class TTSHandler:
    """Handler for text-to-speech functionality"""

    def __init__(self, engine_type: str = 'pyttsx3', config: Dict = None):
        """
        Initialize TTS handler

        Args:
            engine_type: Type of TTS engine ('pyttsx3' or 'gtts')
            config: Configuration dictionary
        """
        self.engine_type = engine_type
        self.config = config or {}
        self.engine = None
        self.is_speaking = False
        self.speech_queue = queue.Queue()
        self.worker_thread = None
        self.stop_worker = False

        # Initialize the selected engine
        self._init_engine()

        # Initialize pygame mixer for audio playback (for gTTS)
        if self.engine_type == 'gtts':
            try:
                pygame.mixer.init()
                print("✅ Pygame mixer initialized for gTTS")
            except Exception as e:
                print(f"⚠️ Could not initialize pygame mixer: {e}")

        # Start the background worker thread
        self._start_worker_thread()

    def _start_worker_thread(self):
        """Start the background worker thread for processing speech queue"""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.stop_worker = False
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            print("✅ TTS worker thread started")

    def _worker_loop(self):
        """Background worker loop to process speech queue"""
        while not self.stop_worker:
            try:
                # Wait for text from queue with timeout
                text = self.speech_queue.get(timeout=0.1)
                if text:
                    self._speak_sync(text)
                self.speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error in TTS worker loop: {e}")
                time.sleep(0.1)

    def _init_engine(self):
        """Initialize the TTS engine"""
        try:
            if self.engine_type == 'pyttsx3':
                self._init_pyttsx3()
            elif self.engine_type == 'gtts':
                self._init_gtts()
            else:
                print(f"❌ Unsupported TTS engine: {self.engine_type}")
                self._init_pyttsx3()  # Fallback to pyttsx3
        except Exception as e:
            print(f"❌ Error initializing TTS engine: {e}")
            # Try fallback
            if self.engine_type != 'pyttsx3':
                try:
                    self._init_pyttsx3()
                except Exception as fallback_error:
                    print(f"❌ Fallback engine also failed: {fallback_error}")
                    self.engine = None
    
    def _init_pyttsx3(self):
        """Initialize pyttsx3 engine"""
        self.engine = pyttsx3.init()
        
        # Set properties from config
        rate = self.config.get('TTS_RATE', 150)
        volume = self.config.get('TTS_VOLUME', 0.8)
        voice_index = self.config.get('TTS_VOICE_INDEX', 0)
        
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        
        # Set voice
        voices = self.engine.getProperty('voices')
        if voices and len(voices) > voice_index:
            self.engine.setProperty('voice', voices[voice_index].id)
        
        print(f"✅ pyttsx3 engine initialized (rate: {rate}, volume: {volume})")
    
    def _init_gtts(self):
        """Initialize gTTS (Google Text-to-Speech)"""
        # gTTS doesn't need initialization, but we'll test connectivity
        try:
            # Test with a simple phrase
            test_text = "Test"
            tts = gTTS(text=test_text, lang='en', slow=False)
            # Don't actually save or play, just test creation
            print("✅ gTTS engine initialized and tested")
            self.engine = 'gtts'  # Use string marker for gTTS
        except Exception as e:
            print(f"❌ gTTS initialization failed: {e}")
            raise e
    
    def speak(self, text: str, async_mode: bool = True) -> bool:
        """
        Convert text to speech

        Args:
            text: Text to speak
            async_mode: Whether to speak asynchronously

        Returns:
            Success status
        """
        if not text or not text.strip():
            print("⚠️ No text provided for TTS")
            return False

        try:
            # Clear any previous items in queue to prevent buildup
            while not self.speech_queue.empty():
                try:
                    self.speech_queue.get_nowait()
                    self.speech_queue.task_done()
                except queue.Empty:
                    break

            if async_mode:
                # Add text to queue for background processing
                self.speech_queue.put(text)
                print(f"📝 Added to TTS queue: '{text}' (queue size: {self.speech_queue.qsize()})")
                return True
            else:
                return self._speak_sync(text)

        except Exception as e:
            print(f"❌ Error in TTS speak: {e}")
            return False
    
    def _speak_sync(self, text: str) -> bool:
        """
        Synchronous speak method
        
        Args:
            text: Text to speak
            
        Returns:
            Success status
        """
        if self.engine is None:
            print("❌ TTS engine not initialized")
            return False
        
        self.is_speaking = True
        
        try:
            if self.engine_type == 'pyttsx3':
                self.engine.say(text)
                self.engine.runAndWait()
                
            elif self.engine_type == 'gtts':
                self._speak_with_gtts(text)
                
            print(f"🗣️ Spoke: '{text}'")
            return True
            
        except Exception as e:
            print(f"❌ Error in sync speak: {e}")
            return False
            
        finally:
            self.is_speaking = False
    
    def _speak_with_gtts(self, text: str):
        """
        Speak using Google Text-to-Speech
        
        Args:
            text: Text to speak
        """
        try:
            # Create gTTS object
            tts = gTTS(text=text, lang='en', slow=False)
            
            # Save to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                temp_path = temp_file.name
                tts.save(temp_path)
            
            # Play the audio file
            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.play()
            
            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except Exception as cleanup_error:
                print(f"⚠️ Could not delete temp file: {cleanup_error}")
                
        except Exception as e:
            print(f"❌ Error in gTTS speak: {e}")
            raise e
    
    def stop_speaking(self):
        """Stop current speech"""
        try:
            if self.engine_type == 'pyttsx3' and self.engine:
                self.engine.stop()
                
            elif self.engine_type == 'gtts':
                pygame.mixer.music.stop()
            
            self.is_speaking = False
            print("🛑 TTS stopped")
            
        except Exception as e:
            print(f"❌ Error stopping TTS: {e}")
    
    def set_rate(self, rate: int):
        """
        Set speaking rate (pyttsx3 only)
        
        Args:
            rate: Speaking rate (words per minute)
        """
        if self.engine_type == 'pyttsx3' and self.engine:
            try:
                self.engine.setProperty('rate', rate)
                print(f"🎚️ TTS rate set to {rate} WPM")
            except Exception as e:
                print(f"❌ Error setting TTS rate: {e}")
    
    def set_volume(self, volume: float):
        """
        Set speaking volume (pyttsx3 only)
        
        Args:
            volume: Volume level (0.0 to 1.0)
        """
        if self.engine_type == 'pyttsx3' and self.engine:
            try:
                volume = max(0.0, min(1.0, volume))  # Clamp between 0 and 1
                self.engine.setProperty('volume', volume)
                print(f"🔊 TTS volume set to {volume}")
            except Exception as e:
                print(f"❌ Error setting TTS volume: {e}")
    
    def set_voice(self, voice_index: int):
        """
        Set voice (pyttsx3 only)
        
        Args:
            voice_index: Index of voice to use
        """
        if self.engine_type == 'pyttsx3' and self.engine:
            try:
                voices = self.engine.getProperty('voices')
                if voices and 0 <= voice_index < len(voices):
                    self.engine.setProperty('voice', voices[voice_index].id)
                    print(f"🗣️ TTS voice set to index {voice_index}")
                else:
                    print(f"⚠️ Voice index {voice_index} out of range")
            except Exception as e:
                print(f"❌ Error setting TTS voice: {e}")
    
    def get_voices(self) -> List[Dict]:
        """
        Get available voices (pyttsx3 only)
        
        Returns:
            List of voice information dictionaries
        """
        if self.engine_type == 'pyttsx3' and self.engine:
            try:
                voices = self.engine.getProperty('voices')
                voice_list = []
                
                for i, voice in enumerate(voices):
                    voice_info = {
                        'index': i,
                        'id': voice.id,
                        'name': voice.name,
                        'gender': getattr(voice, 'gender', 'Unknown'),
                        'age': getattr(voice, 'age', 'Unknown')
                    }
                    voice_list.append(voice_info)
                
                return voice_list
                
            except Exception as e:
                print(f"❌ Error getting voices: {e}")
                return []
        
        return []
    
    def get_properties(self) -> Dict:
        """
        Get current TTS properties
        
        Returns:
            Dictionary of current properties
        """
        properties = {
            'engine_type': self.engine_type,
            'is_speaking': self.is_speaking,
            'available': self.engine is not None
        }
        
        if self.engine_type == 'pyttsx3' and self.engine:
            try:
                properties.update({
                    'rate': self.engine.getProperty('rate'),
                    'volume': self.engine.getProperty('volume'),
                    'voice': self.engine.getProperty('voice'),
                    'voices_count': len(self.engine.getProperty('voices') or [])
                })
            except Exception as e:
                print(f"⚠️ Could not get some properties: {e}")
        
        return properties
    
    def is_available(self) -> bool:
        """
        Check if TTS is available and working
        
        Returns:
            True if TTS is available
        """
        return self.engine is not None
    
    def test_speech(self) -> bool:
        """
        Test TTS functionality
        
        Returns:
            True if test was successful
        """
        test_text = "Hello, this is a test of the text to speech system."
        return self.speak(test_text, async_mode=False)
    
    def cleanup(self):
        """Clean up TTS resources"""
        try:
            # Stop the worker thread
            self.stop_worker = True
            if self.worker_thread and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=1.0)

            self.stop_speaking()

            if self.engine_type == 'pyttsx3' and self.engine:
                # pyttsx3 doesn't have explicit cleanup
                pass

            elif self.engine_type == 'gtts':
                pygame.mixer.quit()

            self.engine = None
            print("🧹 TTS handler cleaned up")

        except Exception as e:
            print(f"⚠️ Error during TTS cleanup: {e}")

# Utility function for quick TTS
def quick_speak(text: str, engine_type: str = 'pyttsx3') -> bool:
    """
    Quick function to speak text without creating a persistent handler
    
    Args:
        text: Text to speak
        engine_type: TTS engine to use
        
    Returns:
        Success status
    """
    try:
        tts = TTSHandler(engine_type=engine_type)
        result = tts.speak(text, async_mode=False)
        tts.cleanup()
        return result
    except Exception as e:
        print(f"❌ Quick speak error: {e}")
        return False