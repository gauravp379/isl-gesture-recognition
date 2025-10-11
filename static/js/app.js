/**
 * ISL Gesture Recognition System - Frontend JavaScript
 */

class ISLGestureApp {
    constructor() {
        this.currentSentence = '';
        this.isProcessing = false;
        this.fpsCounter = 0;
        this.performanceData = {};
        
        // Initialize the application
        this.init();
    }

    /**
     * Initialize the application
     */
    init() {
        console.log('🚀 Initializing ISL Gesture Recognition System...');
        
        // Bind event listeners
        this.bindEventListeners();
        
        // Start real-time data updates
        this.startDataUpdates();
        
        // Initialize UI components
        this.initializeUI();
        
        console.log('✅ Application initialized successfully');
    }

    /**
     * Bind event listeners to DOM elements
     */
    bindEventListeners() {
        // Speak button
        document.getElementById('speak-btn').addEventListener('click', () => {
            this.speakText();
        });

        // Clear button
        document.getElementById('clear-btn').addEventListener('click', () => {
            this.clearSentence();
        });

        // Video stream error handling
        const videoStream = document.getElementById('video-stream');
        videoStream.addEventListener('error', () => {
            this.handleVideoError();
        });

        videoStream.addEventListener('load', () => {
            this.handleVideoLoad();
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (event) => {
            this.handleKeyboardShortcuts(event);
        });
    }

    /**
     * Start real-time data updates
     */
    startDataUpdates() {
        // Update gesture data every 100ms
        setInterval(() => {
            this.updateGestureData();
        }, 100);

        // Update performance metrics every 1 second
        setInterval(() => {
            this.updatePerformanceMetrics();
        }, 1000);

        // Update FPS counter every 500ms
        setInterval(() => {
            this.updateFPSCounter();
        }, 500);
    }

    /**
     * Initialize UI components
     */
    initializeUI() {
        // Add loading states
        this.showLoadingState('Initializing camera...');
        
        // Hide loading after 3 seconds
        setTimeout(() => {
            this.hideLoadingState();
        }, 3000);

        // Initialize tooltips
        this.initializeTooltips();
    }

    /**
     * Update gesture recognition data
     */
    async updateGestureData() {
        if (this.isProcessing) return;

        try {
            this.isProcessing = true;
            const response = await fetch('/api/gesture_data');
            const data = await response.json();

            if (data.error) {
                console.error('Error fetching gesture data:', data.error);
                return;
            }

            // Update current sentence
            if (data.current_sentence !== this.currentSentence) {
                this.updateSentenceDisplay(data.current_sentence);
                this.updateWordSuggestions(data.current_sentence);
            }

            // Update confidence if available
            if (data.confidence !== undefined) {
                this.updateConfidence(data.confidence);
            }

        } catch (error) {
            console.error('Error updating gesture data:', error);
        } finally {
            this.isProcessing = false;
        }
    }

    /**
     * Update performance metrics
     */
    async updatePerformanceMetrics() {
        try {
            const response = await fetch('/api/performance');
            const data = await response.json();

            if (!data.error) {
                this.performanceData = data;
                this.displayPerformanceMetrics(data);
            }

        } catch (error) {
            console.error('Error updating performance metrics:', error);
        }
    }

    /**
     * Update FPS counter
     */
    updateFPSCounter() {
        const fpsElement = document.getElementById('fps-counter');
        if (this.performanceData.fps) {
            fpsElement.textContent = Math.round(this.performanceData.fps);
            fpsElement.className = this.performanceData.fps > 25 ? 'text-success' : 'text-warning';
        }
    }

    /**
     * Update sentence display
     */
    updateSentenceDisplay(sentence) {
        this.currentSentence = sentence;
        const display = document.getElementById('sentence-display');
        display.value = sentence;
        
        // Add animation
        display.classList.add('slide-up');
        setTimeout(() => {
            display.classList.remove('slide-up');
        }, 300);

        // Extract and display current character
        if (sentence.length > 0) {
            const lastChar = sentence.charAt(sentence.length - 1);
            this.updateCurrentCharacter(lastChar);
        }
    }

    /**
     * Update current character display
     */
    updateCurrentCharacter(character) {
        const element = document.getElementById('current-character');
        element.textContent = character.toUpperCase();
        element.classList.add('pulse');
        
        setTimeout(() => {
            element.classList.remove('pulse');
        }, 500);
    }

    /**
     * Update confidence bar
     */
    updateConfidence(confidence) {
        const confidenceBar = document.getElementById('confidence-bar');
        const confidenceText = document.getElementById('confidence-text');
        
        const percentage = Math.round(confidence * 100);
        confidenceBar.style.width = percentage + '%';
        confidenceText.textContent = percentage + '%';
        
        // Change color based on confidence level
        confidenceBar.className = 'progress-bar ' + 
            (percentage >= 80 ? 'bg-success' : 
             percentage >= 60 ? 'bg-warning' : 'bg-danger');
    }

    /**
     * Update word suggestions
     */
    updateWordSuggestions(sentence) {
        const suggestionsContainer = document.getElementById('suggestions');
        const words = this.generateWordSuggestions(sentence);
        
        suggestionsContainer.innerHTML = '';
        words.forEach(word => {
            const button = document.createElement('button');
            button.className = 'suggestion-btn';
            button.textContent = word;
            button.onclick = () => this.applySuggestion(word);
            suggestionsContainer.appendChild(button);
        });
    }

    /**
     * Generate word suggestions based on current sentence
     */
    generateWordSuggestions(sentence) {
        // Simple word suggestion logic
        const commonWords = [
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER',
            'HAVE', 'THERE', 'SAY', 'EACH', 'WHICH', 'SHE', 'HOW', 'WILL', 'MY',
            'HELLO', 'THANK', 'PLEASE', 'HELP', 'GOOD', 'YES', 'NO', 'WATER'
        ];
        
        if (!sentence || sentence.length < 2) {
            return commonWords.slice(0, 6);
        }
        
        // Filter words that start with the last few characters
        const lastWord = sentence.split(' ').pop().toUpperCase();
        const suggestions = commonWords
            .filter(word => word.startsWith(lastWord) && word !== lastWord)
            .slice(0, 6);
            
        return suggestions.length > 0 ? suggestions : commonWords.slice(0, 6);
    }

    /**
     * Apply word suggestion
     */
    applySuggestion(word) {
        const words = this.currentSentence.split(' ');
        words[words.length - 1] = word;
        const newSentence = words.join(' ') + ' ';
        
        this.updateSentenceDisplay(newSentence);
        
        // Send to backend
        this.sendSentenceUpdate(newSentence);
    }

    /**
     * Display performance metrics
     */
    displayPerformanceMetrics(data) {
        // Update average FPS
        const avgFpsElement = document.getElementById('avg-fps');
        if (data.avg_fps !== undefined) {
            avgFpsElement.textContent = Math.round(data.avg_fps);
        }

        // Update accuracy
        const accuracyElement = document.getElementById('accuracy');
        if (data.accuracy !== undefined) {
            accuracyElement.textContent = Math.round(data.accuracy * 100) + '%';
        }

        // Update latency
        const latencyElement = document.getElementById('latency');
        if (data.latency !== undefined) {
            latencyElement.textContent = Math.round(data.latency) + 'ms';
        }
    }

    /**
     * Speak current text
     */
    async speakText() {
        const speakBtn = document.getElementById('speak-btn');
        const originalText = speakBtn.innerHTML;
        
        try {
            speakBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Speaking...';
            speakBtn.disabled = true;

            const response = await fetch('/api/speak', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: this.currentSentence
                })
            });

            const result = await response.json();
            
            if (!result.success) {
                throw new Error(result.error || 'Failed to speak text');
            }

            // Show success feedback
            speakBtn.innerHTML = '<i class="fas fa-check me-2"></i>Spoken!';
            setTimeout(() => {
                speakBtn.innerHTML = originalText;
            }, 2000);

        } catch (error) {
            console.error('Error speaking text:', error);
            alert('Error: ' + error.message);
            speakBtn.innerHTML = originalText;
        } finally {
            speakBtn.disabled = false;
        }
    }

    /**
     * Clear current sentence
     */
    async clearSentence() {
        try {
            const response = await fetch('/api/clear_sentence', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            const result = await response.json();
            
            if (result.success) {
                this.updateSentenceDisplay('');
                this.updateCurrentCharacter('--');
                this.updateConfidence(0);
                this.updateWordSuggestions('');
            }

        } catch (error) {
            console.error('Error clearing sentence:', error);
        }
    }

    /**
     * Handle video stream errors
     */
    handleVideoError() {
        const statusElement = document.getElementById('camera-status');
        statusElement.innerHTML = '<i class="fas fa-circle text-danger"></i> Camera Error';
        console.error('Video stream error');
    }

    /**
     * Handle video stream load
     */
    handleVideoLoad() {
        const statusElement = document.getElementById('camera-status');
        statusElement.innerHTML = '<i class="fas fa-circle text-success"></i> Camera Active';
    }

    /**
     * Handle keyboard shortcuts
     */
    handleKeyboardShortcuts(event) {
        // Ctrl/Cmd + Space: Speak
        if ((event.ctrlKey || event.metaKey) && event.code === 'Space') {
            event.preventDefault();
            this.speakText();
        }
        
        // Ctrl/Cmd + Delete: Clear
        if ((event.ctrlKey || event.metaKey) && event.code === 'Backspace') {
            event.preventDefault();
            this.clearSentence();
        }
    }

    /**
     * Show loading state
     */
    showLoadingState(message = 'Loading...') {
        // Implementation for loading state
        console.log('Loading:', message);
    }

    /**
     * Hide loading state
     */
    hideLoadingState() {
        // Implementation for hiding loading state
        console.log('Loading complete');
    }

    /**
     * Initialize tooltips
     */
    initializeTooltips() {
        // Add tooltips to buttons
        document.getElementById('speak-btn').title = 'Speak text (Ctrl+Space)';
        document.getElementById('clear-btn').title = 'Clear text (Ctrl+Backspace)';
    }

    /**
     * Send sentence update to backend
     */
    async sendSentenceUpdate(sentence) {
        // Implementation to sync sentence with backend if needed
        console.log('Sentence updated:', sentence);
    }

    /**
     * Check application health
     */
    async checkHealth() {
        try {
            const response = await fetch('/health');
            const health = await response.json();
            console.log('Application health:', health);
            return health;
        } catch (error) {
            console.error('Health check failed:', error);
            return null;
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.islApp = new ISLGestureApp();
    
    // Perform initial health check
    window.islApp.checkHealth().then(health => {
        if (health && health.status === 'healthy') {
            console.log('✅ Application is healthy and ready!');
        } else {
            console.warn('⚠️ Application health check failed');
        }
    });
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
        console.log('Page became visible - resuming updates');
        // Resume updates if needed
    } else {
        console.log('Page became hidden - pausing updates');
        // Pause updates to save resources
    }
});

// Handle connection errors
window.addEventListener('online', () => {
    console.log('Connection restored');
});

window.addEventListener('offline', () => {
    console.log('Connection lost');
});