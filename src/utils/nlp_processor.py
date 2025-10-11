"""
Natural Language Processing Module for ISL Gesture Recognition
Processes gesture sequences to form meaningful sentences
"""
import re
from typing import List, Dict, Optional, Tuple
import nltk
from collections import Counter, deque
import spacy
import time

class NLPProcessor:
    """NLP processor for gesture sequences"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize NLP processor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.gesture_buffer = deque(maxlen=50)  # Store last 50 gestures
        self.word_buffer = deque(maxlen=20)     # Store last 20 words
        self.sentence_buffer = ""
        
        # Initialize language models
        self._init_language_models()
        
        # Load dictionaries and rules
        self._load_dictionaries()
        
        # Gesture timing
        self.last_gesture_time = 0
        self.gesture_timeout = 2.0  # seconds
        
        print("✅ NLP Processor initialized")
    
    def _init_language_models(self):
        """Initialize NLTK and spaCy models"""
        try:
            # Download required NLTK data
            nltk_downloads = [
                'punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
                'omw-1.4', 'vader_lexicon'
            ]
            
            for item in nltk_downloads:
                try:
                    nltk.data.find(f'tokenizers/{item}')
                except LookupError:
                    try:
                        nltk.download(item, quiet=True)
                    except Exception as e:
                        print(f"⚠️ Could not download {item}: {e}")
            
            # Load spaCy model (fallback if not available)
            try:
                self.nlp = spacy.load("en_core_web_sm")
                print("✅ spaCy model loaded")
            except OSError:
                print("⚠️ spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
                
        except Exception as e:
            print(f"⚠️ Error initializing language models: {e}")
            self.nlp = None
    
    def _load_dictionaries(self):
        """Load dictionaries and language rules"""
        # Common English words dictionary
        self.common_words = {
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER',
            'WAS', 'ONE', 'OUR', 'HAD', 'BY', 'HOT', 'WORD', 'WHAT', 'SOME', 'TIME',
            'VERY', 'WHEN', 'WHERE', 'RIGHT', 'SHE', 'EACH', 'WHICH', 'DO', 'HOW',
            'IF', 'WILL', 'UP', 'OTHER', 'ABOUT', 'OUT', 'MANY', 'THEN', 'THEM',
            'THESE', 'SO', 'HIM', 'HIS', 'HAS', 'MORE', 'HE', 'MY', 'NOW', 'LOOK',
            'ONLY', 'COME', 'ITS', 'OVER', 'THINK', 'ALSO', 'BACK', 'AFTER', 'USE',
            'TWO', 'WAY', 'EVEN', 'NEW', 'WANT', 'BECAUSE', 'ANY', 'GIVE', 'DAY',
            'MOST', 'US', 'IS', 'WATER', 'LONG', 'GET', 'HERE', 'MAKE', 'THING',
            'SEE', 'WORK', 'LIFE', 'CALL', 'GOOD', 'MOVE', 'LIVE', 'WHERE', 'MUCH',
            'TAKE', 'WHY', 'HELP', 'PUT', 'DIFFERENT', 'AWAY', 'AGAIN', 'OFF', 'WENT',
            'OLD', 'TELL', 'SAY', 'GREAT', 'MAN', 'BIG', 'GROUP', 'ASK', 'NEED',
            'TRY', 'TURN', 'STILL', 'PLACE', 'PART', 'WORLD', 'OVER', 'SUCH', 'GO'
        }
        
        # ISL-specific words and phrases
        self.isl_words = {
            'HELLO', 'GOODBYE', 'THANK', 'PLEASE', 'SORRY', 'YES', 'NO', 'HELP',
            'WATER', 'FOOD', 'TOILET', 'HOME', 'SCHOOL', 'WORK', 'FAMILY',
            'MOTHER', 'FATHER', 'BROTHER', 'SISTER', 'FRIEND', 'TEACHER',
            'DOCTOR', 'HOSPITAL', 'MEDICINE', 'PAIN', 'HAPPY', 'SAD', 'ANGRY',
            'LOVE', 'LIKE', 'WANT', 'NEED', 'HAVE', 'GO', 'COME', 'SIT', 'STAND'
        }
        
        # Word completion patterns
        self.word_patterns = {
            'TH': ['THE', 'THAT', 'THEY', 'THEM', 'THINK', 'THANK', 'THING'],
            'HE': ['HELP', 'HELLO', 'HER', 'HERE', 'HE'],
            'WH': ['WHAT', 'WHERE', 'WHEN', 'WHO', 'WHY', 'WHICH'],
            'GO': ['GOOD', 'GO', 'GOT'],
            'WA': ['WATER', 'WANT', 'WAS', 'WAY'],
            'PL': ['PLEASE', 'PLACE', 'PLAY'],
            'WO': ['WORK', 'WORLD', 'WORD', 'WORRY']
        }
        
        # Common ISL sentence starters
        self.sentence_starters = [
            'I', 'YOU', 'WE', 'THEY', 'THIS', 'THAT', 'PLEASE', 'CAN', 'WILL',
            'WHAT', 'WHERE', 'WHEN', 'HOW', 'WHY'
        ]
        
        # Punctuation rules
        self.sentence_enders = {'.', '!', '?'}
        
        print("📚 Dictionaries and rules loaded")
    
    def process_gestures(self, gestures: List[str]) -> str:
        """
        Process a sequence of gestures to form text
        
        Args:
            gestures: List of gesture characters
            
        Returns:
            Processed text string
        """
        if not gestures:
            return self.sentence_buffer
        
        current_time = time.time()
        
        # Add gestures to buffer
        for gesture in gestures:
            if gesture and gesture.strip():
                self.gesture_buffer.append({
                    'character': gesture.upper(),
                    'timestamp': current_time
                })
        
        # Process the buffer to form words and sentences
        return self._process_buffer()
    
    def _process_buffer(self) -> str:
        """
        Process the gesture buffer to form meaningful text
        
        Returns:
            Processed text string
        """
        if not self.gesture_buffer:
            return self.sentence_buffer
        
        # Convert gestures to raw text
        raw_text = ''.join([g['character'] for g in self.gesture_buffer])
        
        # Clean and process text
        processed_text = self._clean_text(raw_text)
        processed_text = self._form_words(processed_text)
        processed_text = self._form_sentences(processed_text)
        
        return processed_text
    
    def _clean_text(self, text: str) -> str:
        """
        Clean raw gesture text
        
        Args:
            text: Raw text from gestures
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove duplicate consecutive characters
        cleaned = re.sub(r'(.)\1{2,}', r'\1', text)
        
        # Remove non-alphabetic characters (keeping spaces and numbers)
        cleaned = re.sub(r'[^A-Z0-9\s]', '', cleaned)
        
        # Clean up multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def _form_words(self, text: str) -> str:
        """
        Form words from character sequences
        
        Args:
            text: Cleaned character sequence
            
        Returns:
            Text with word boundaries
        """
        if not text:
            return ""
        
        words = []
        current_word = ""
        
        for char in text:
            if char == ' ':
                if current_word:
                    # Process the current word
                    processed_word = self._complete_word(current_word)
                    if processed_word:
                        words.append(processed_word)
                    current_word = ""
            else:
                current_word += char
        
        # Process the last word
        if current_word:
            processed_word = self._complete_word(current_word)
            if processed_word:
                words.append(processed_word)
        
        return ' '.join(words)
    
    def _complete_word(self, partial_word: str) -> Optional[str]:
        """
        Complete a partial word using dictionaries and patterns
        
        Args:
            partial_word: Partial word to complete
            
        Returns:
            Completed word or None
        """
        if not partial_word:
            return None
        
        partial_upper = partial_word.upper()
        
        # If it's already a complete word, return it
        if partial_upper in self.common_words or partial_upper in self.isl_words:
            return partial_upper
        
        # Check if it's a number
        if partial_word.isdigit():
            return partial_word
        
        # Try to complete using patterns
        for pattern, completions in self.word_patterns.items():
            if partial_upper.startswith(pattern):
                # Return the most likely completion
                for completion in completions:
                    if completion.startswith(partial_upper):
                        return completion
        
        # If partial word is long enough and matches common words
        if len(partial_upper) >= 2:
            candidates = []
            
            # Check common words
            for word in self.common_words:
                if word.startswith(partial_upper):
                    candidates.append(word)
            
            # Check ISL words
            for word in self.isl_words:
                if word.startswith(partial_upper):
                    candidates.append(word)
            
            # Return the shortest candidate (most likely)
            if candidates:
                return min(candidates, key=len)
        
        # If we can't complete it, return the partial word
        return partial_upper if len(partial_upper) >= 2 else None
    
    def _form_sentences(self, text: str) -> str:
        """
        Form proper sentences from words
        
        Args:
            text: Text with word boundaries
            
        Returns:
            Text with proper sentence structure
        """
        if not text:
            return self.sentence_buffer
        
        words = text.split()
        if not words:
            return self.sentence_buffer
        
        # Add to word buffer
        for word in words:
            self.word_buffer.append(word)
        
        # Form sentences from word buffer
        sentence_words = list(self.word_buffer)
        
        # Basic sentence formation
        if len(sentence_words) >= 2:
            sentence = ' '.join(sentence_words)
            
            # Add punctuation if needed
            if not sentence.endswith(('.', '!', '?')):
                # Simple heuristic for punctuation
                if any(q in sentence for q in ['WHAT', 'WHERE', 'WHEN', 'HOW', 'WHY']):
                    sentence += '?'
                elif any(e in sentence for e in ['HELLO', 'THANK', 'PLEASE', 'HELP']):
                    sentence += '!'
                else:
                    sentence += '.'
            
            return sentence
        
        return ' '.join(sentence_words)
    
    def get_word_suggestions(self, partial_text: str, count: int = 5) -> List[str]:
        """
        Get word suggestions based on partial text
        
        Args:
            partial_text: Partial text to get suggestions for
            count: Number of suggestions to return
            
        Returns:
            List of suggested words
        """
        if not partial_text:
            return list(self.common_words)[:count]
        
        words = partial_text.upper().split()
        if not words:
            return list(self.common_words)[:count]
        
        last_word = words[-1]
        suggestions = []
        
        # Get completions for the last word
        all_words = list(self.common_words) + list(self.isl_words)
        
        for word in all_words:
            if word.startswith(last_word) and word != last_word:
                suggestions.append(word)
        
        # Sort by word length (prefer shorter words)
        suggestions.sort(key=len)
        
        return suggestions[:count]
    
    def clear_buffers(self):
        """Clear all buffers"""
        self.gesture_buffer.clear()
        self.word_buffer.clear()
        self.sentence_buffer = ""
        print("🧹 NLP buffers cleared")
    
    def get_statistics(self) -> Dict:
        """
        Get processing statistics
        
        Returns:
            Dictionary of statistics
        """
        return {
            'gesture_buffer_size': len(self.gesture_buffer),
            'word_buffer_size': len(self.word_buffer),
            'sentence_length': len(self.sentence_buffer.split()) if self.sentence_buffer else 0,
            'common_words_count': len(self.common_words),
            'isl_words_count': len(self.isl_words),
            'patterns_count': len(self.word_patterns)
        }
    
    def add_custom_word(self, word: str):
        """
        Add a custom word to the ISL dictionary
        
        Args:
            word: Word to add
        """
        if word and word.strip():
            self.isl_words.add(word.upper().strip())
            print(f"📝 Added custom word: {word}")
    
    def add_custom_pattern(self, prefix: str, completions: List[str]):
        """
        Add a custom completion pattern
        
        Args:
            prefix: Prefix pattern
            completions: List of possible completions
        """
        if prefix and completions:
            self.word_patterns[prefix.upper()] = [c.upper() for c in completions]
            print(f"📝 Added custom pattern: {prefix} -> {completions}")
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze text using NLP techniques
        
        Args:
            text: Text to analyze
            
        Returns:
            Analysis results
        """
        if not text:
            return {}
        
        analysis = {
            'word_count': len(text.split()),
            'character_count': len(text),
            'sentence_count': len([s for s in text.split('.') if s.strip()]),
        }
        
        # Use spaCy if available
        if self.nlp:
            try:
                doc = self.nlp(text)
                analysis.update({
                    'entities': [(ent.text, ent.label_) for ent in doc.ents],
                    'pos_tags': [(token.text, token.pos_) for token in doc],
                    'sentiment': 'positive' if 'GOOD' in text or 'HAPPY' in text else 'neutral'
                })
            except Exception as e:
                print(f"⚠️ Error in spaCy analysis: {e}")
        
        return analysis