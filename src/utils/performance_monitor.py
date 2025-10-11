"""
Performance Monitoring Module for ISL Gesture Recognition System
Tracks system performance metrics including FPS, accuracy, and latency
"""
import time
import psutil
from typing import Dict, List, Optional
from collections import deque
import statistics
import json
import os
from datetime import datetime, timedelta

class PerformanceMonitor:
    """Monitor system performance metrics"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize performance monitor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Performance metrics storage
        self.fps_history = deque(maxlen=1000)
        self.accuracy_history = deque(maxlen=1000)
        self.latency_history = deque(maxlen=1000)
        self.confidence_history = deque(maxlen=1000)
        
        # System metrics
        self.cpu_history = deque(maxlen=100)
        self.memory_history = deque(maxlen=100)
        self.gpu_history = deque(maxlen=100)
        
        # Timing
        self.start_time = time.time()
        self.last_fps_calculation = time.time()
        self.frame_count = 0
        self.current_fps = 0
        
        # Logging
        self.log_enabled = self.config.get('LOG_PERFORMANCE', True)
        self.log_file = self.config.get('LOG_FILE', 'logs/performance.log')
        self.log_interval = self.config.get('FPS_LOG_INTERVAL', 30)  # seconds
        self.last_log_time = time.time()
        
        # Performance thresholds
        self.fps_threshold = 20.0
        self.accuracy_threshold = 0.8
        self.latency_threshold = 100.0  # milliseconds
        self.cpu_threshold = 80.0  # percentage
        self.memory_threshold = 80.0  # percentage
        
        # Alerts
        self.alerts = deque(maxlen=50)
        
        print("✅ Performance monitor initialized")
    
    def log_fps(self, fps: float):
        """
        Log FPS measurement
        
        Args:
            fps: Frames per second
        """
        self.fps_history.append({
            'value': fps,
            'timestamp': time.time()
        })
        self.current_fps = fps
        
        # Check threshold
        if fps < self.fps_threshold:
            self._add_alert(f"Low FPS detected: {fps:.1f}", 'warning')
    
    def log_accuracy(self, accuracy: float):
        """
        Log accuracy measurement
        
        Args:
            accuracy: Model accuracy (0.0 to 1.0)
        """
        self.accuracy_history.append({
            'value': accuracy,
            'timestamp': time.time()
        })
        
        # Check threshold
        if accuracy < self.accuracy_threshold:
            self._add_alert(f"Low accuracy detected: {accuracy:.2f}", 'warning')
    
    def log_latency(self, latency: float):
        """
        Log latency measurement
        
        Args:
            latency: Prediction latency in milliseconds
        """
        self.latency_history.append({
            'value': latency,
            'timestamp': time.time()
        })
        
        # Check threshold
        if latency > self.latency_threshold:
            self._add_alert(f"High latency detected: {latency:.1f}ms", 'warning')
    
    def log_confidence(self, confidence: float):
        """
        Log confidence score
        
        Args:
            confidence: Model confidence (0.0 to 1.0)
        """
        self.confidence_history.append({
            'value': confidence,
            'timestamp': time.time()
        })
    
    def update_system_metrics(self):
        """Update system resource metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.cpu_history.append({
                'value': cpu_percent,
                'timestamp': time.time()
            })
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.memory_history.append({
                'value': memory_percent,
                'timestamp': time.time()
            })
            
            # Check thresholds
            if cpu_percent > self.cpu_threshold:
                self._add_alert(f"High CPU usage: {cpu_percent:.1f}%", 'warning')
            
            if memory_percent > self.memory_threshold:
                self._add_alert(f"High memory usage: {memory_percent:.1f}%", 'warning')
            
            # Try to get GPU info (if available)
            try:
                # This would require additional libraries like nvidia-ml-py
                # For now, we'll skip GPU monitoring
                pass
            except Exception:
                pass
                
        except Exception as e:
            print(f"⚠️ Error updating system metrics: {e}")
    
    def calculate_frame_rate(self):
        """Calculate current frame rate"""
        current_time = time.time()
        self.frame_count += 1
        
        # Calculate FPS every second
        if current_time - self.last_fps_calculation >= 1.0:
            fps = self.frame_count / (current_time - self.last_fps_calculation)
            self.log_fps(fps)
            
            # Reset counters
            self.frame_count = 0
            self.last_fps_calculation = current_time
    
    def get_current_fps(self) -> float:
        """
        Get current FPS
        
        Returns:
            Current frames per second
        """
        return self.current_fps
    
    def get_average_fps(self, duration: int = 60) -> float:
        """
        Get average FPS over duration
        
        Args:
            duration: Duration in seconds
            
        Returns:
            Average FPS
        """
        if not self.fps_history:
            return 0.0
        
        current_time = time.time()
        cutoff_time = current_time - duration
        
        recent_fps = [
            entry['value'] for entry in self.fps_history 
            if entry['timestamp'] >= cutoff_time
        ]
        
        return statistics.mean(recent_fps) if recent_fps else 0.0
    
    def get_average_accuracy(self, duration: int = 300) -> float:
        """
        Get average accuracy over duration
        
        Args:
            duration: Duration in seconds
            
        Returns:
            Average accuracy
        """
        if not self.accuracy_history:
            return 0.0
        
        current_time = time.time()
        cutoff_time = current_time - duration
        
        recent_accuracy = [
            entry['value'] for entry in self.accuracy_history 
            if entry['timestamp'] >= cutoff_time
        ]
        
        return statistics.mean(recent_accuracy) if recent_accuracy else 0.0
    
    def get_average_latency(self, duration: int = 60) -> float:
        """
        Get average latency over duration
        
        Args:
            duration: Duration in seconds
            
        Returns:
            Average latency in milliseconds
        """
        if not self.latency_history:
            return 0.0
        
        current_time = time.time()
        cutoff_time = current_time - duration
        
        recent_latency = [
            entry['value'] for entry in self.latency_history 
            if entry['timestamp'] >= cutoff_time
        ]
        
        return statistics.mean(recent_latency) if recent_latency else 0.0
    
    def get_system_metrics(self) -> Dict:
        """
        Get current system metrics
        
        Returns:
            Dictionary of system metrics
        """
        cpu_current = self.cpu_history[-1]['value'] if self.cpu_history else 0
        memory_current = self.memory_history[-1]['value'] if self.memory_history else 0
        
        return {
            'cpu_percent': cpu_current,
            'memory_percent': memory_current,
            'cpu_average': statistics.mean([m['value'] for m in self.cpu_history]) if self.cpu_history else 0,
            'memory_average': statistics.mean([m['value'] for m in self.memory_history]) if self.memory_history else 0,
            'uptime_seconds': time.time() - self.start_time
        }
    
    def get_metrics(self) -> Dict:
        """
        Get all performance metrics
        
        Returns:
            Dictionary of all metrics
        """
        return {
            'fps': self.get_current_fps(),
            'avg_fps': self.get_average_fps(),
            'accuracy': self.get_average_accuracy(),
            'latency': self.get_average_latency(),
            'system': self.get_system_metrics(),
            'total_predictions': len(self.accuracy_history),
            'uptime': time.time() - self.start_time,
            'alerts_count': len(self.alerts)
        }
    
    def get_performance_summary(self) -> Dict:
        """
        Get performance summary report
        
        Returns:
            Summary report dictionary
        """
        fps_values = [entry['value'] for entry in self.fps_history]
        accuracy_values = [entry['value'] for entry in self.accuracy_history]
        latency_values = [entry['value'] for entry in self.latency_history]
        confidence_values = [entry['value'] for entry in self.confidence_history]
        
        summary = {
            'period': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'duration_hours': (time.time() - self.start_time) / 3600
            },
            'fps': self._get_metric_summary(fps_values),
            'accuracy': self._get_metric_summary(accuracy_values),
            'latency': self._get_metric_summary(latency_values),
            'confidence': self._get_metric_summary(confidence_values),
            'system': self.get_system_metrics(),
            'alerts': list(self.alerts)
        }
        
        return summary
    
    def _get_metric_summary(self, values: List[float]) -> Dict:
        """
        Get statistical summary of metric values
        
        Args:
            values: List of metric values
            
        Returns:
            Statistical summary
        """
        if not values:
            return {
                'count': 0,
                'mean': 0,
                'min': 0,
                'max': 0,
                'std': 0
            }
        
        return {
            'count': len(values),
            'mean': statistics.mean(values),
            'min': min(values),
            'max': max(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0
        }
    
    def _add_alert(self, message: str, severity: str = 'info'):
        """
        Add performance alert
        
        Args:
            message: Alert message
            severity: Alert severity ('info', 'warning', 'error')
        """
        alert = {
            'message': message,
            'severity': severity,
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat()
        }
        
        self.alerts.append(alert)
        print(f"🚨 {severity.upper()}: {message}")
        
        # Log to file if enabled
        if self.log_enabled:
            self._log_to_file(f"ALERT [{severity}]: {message}")
    
    def _log_to_file(self, message: str):
        """
        Log message to performance log file
        
        Args:
            message: Message to log
        """
        if not self.log_enabled:
            return
        
        try:
            # Ensure log directory exists
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            
            timestamp = datetime.now().isoformat()
            log_entry = f"[{timestamp}] {message}\n"
            
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
                
        except Exception as e:
            print(f"⚠️ Error writing to log file: {e}")
    
    def log_periodic_metrics(self):
        """Log metrics periodically"""
        current_time = time.time()
        
        if current_time - self.last_log_time >= self.log_interval:
            metrics = self.get_metrics()
            
            log_message = (
                f"FPS: {metrics['fps']:.1f} | "
                f"Avg FPS: {metrics['avg_fps']:.1f} | "
                f"Accuracy: {metrics['accuracy']:.2f} | "
                f"Latency: {metrics['latency']:.1f}ms | "
                f"CPU: {metrics['system']['cpu_percent']:.1f}% | "
                f"Memory: {metrics['system']['memory_percent']:.1f}%"
            )
            
            self._log_to_file(log_message)
            self.last_log_time = current_time
    
    def export_metrics(self, filepath: str):
        """
        Export metrics to JSON file
        
        Args:
            filepath: Path to export file
        """
        try:
            metrics_data = {
                'summary': self.get_performance_summary(),
                'fps_history': list(self.fps_history),
                'accuracy_history': list(self.accuracy_history),
                'latency_history': list(self.latency_history),
                'confidence_history': list(self.confidence_history),
                'cpu_history': list(self.cpu_history),
                'memory_history': list(self.memory_history),
                'alerts': list(self.alerts)
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=2, default=str)
            
            print(f"📊 Metrics exported to {filepath}")
            
        except Exception as e:
            print(f"❌ Error exporting metrics: {e}")
    
    def clear_history(self):
        """Clear all performance history"""
        self.fps_history.clear()
        self.accuracy_history.clear()
        self.latency_history.clear()
        self.confidence_history.clear()
        self.cpu_history.clear()
        self.memory_history.clear()
        self.alerts.clear()
        
        print("🧹 Performance history cleared")
    
    def get_health_status(self) -> Dict:
        """
        Get overall system health status
        
        Returns:
            Health status dictionary
        """
        current_metrics = self.get_metrics()
        
        # Determine health based on thresholds
        fps_healthy = current_metrics['fps'] >= self.fps_threshold
        accuracy_healthy = current_metrics['accuracy'] >= self.accuracy_threshold
        latency_healthy = current_metrics['latency'] <= self.latency_threshold
        cpu_healthy = current_metrics['system']['cpu_percent'] <= self.cpu_threshold
        memory_healthy = current_metrics['system']['memory_percent'] <= self.memory_threshold
        
        overall_healthy = all([fps_healthy, accuracy_healthy, latency_healthy, cpu_healthy, memory_healthy])
        
        return {
            'overall_status': 'healthy' if overall_healthy else 'warning',
            'fps_healthy': fps_healthy,
            'accuracy_healthy': accuracy_healthy,
            'latency_healthy': latency_healthy,
            'cpu_healthy': cpu_healthy,
            'memory_healthy': memory_healthy,
            'recent_alerts': len([a for a in self.alerts if time.time() - a['timestamp'] < 300]),  # Last 5 minutes
            'uptime_hours': (time.time() - self.start_time) / 3600
        }
    
    def cleanup(self):
        """Clean up performance monitor"""
        # Final log
        if self.log_enabled:
            self._log_to_file("Performance monitor shutting down")
        
        print("🧹 Performance monitor cleaned up")