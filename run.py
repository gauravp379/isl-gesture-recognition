#!/usr/bin/env python3
"""
ISL Gesture Recognition System Launcher
Simple script to start the application with proper error handling
"""
import sys
import os

def main():
    """Main launcher function"""
    print(" Starting ISL Gesture Recognition System...")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print(" Python 3.7 or higher is required")
        print(f"   Current version: {sys.version}")
        sys.exit(1)
    
    print(f" Python version: {sys.version}")
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print(" app.py not found. Please run this script from the project root directory.")
        sys.exit(1)
    
    # Check if virtual environment is activated (optional)
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print(" Virtual environment detected")
    else:
        print(" No virtual environment detected. Consider using one for better dependency management.")
    
    try:
        # Import and run the main application
        print("\n Loading application modules...")
        
        # Check critical imports first
        critical_modules = [
            'flask', 'cv2', 'mediapipe', 'numpy', 'tensorflow'
        ]
        
        missing_modules = []
        for module in critical_modules:
            try:
                __import__(module)
                print(f" {module}")
            except ImportError:
                missing_modules.append(module)
                print(f" {module} - NOT FOUND")
        
        if missing_modules:
            print(f"\n Missing required modules: {', '.join(missing_modules)}")
            print(" Install them with: pip install -r requirements.txt")
            sys.exit(1)
        
        print("\n Starting Flask server...")
        print(" Once started, open your browser and go to: http://localhost:5000")
        print(" Press Ctrl+C to stop the server")
        print("=" * 60)
        
        # Import and run the app
        from app import app
        
        # Run the Flask application
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            threaded=True
        )
        
    except ImportError as e:
        print(f" Import Error: {e}")
        print("d Make sure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n Server stopped by user")
        print(" Thank you for using ISL Gesture Recognition System!")
        
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        print("Please check your configuration and try again")
        sys.exit(1)

if __name__ == "__main__":
    main()