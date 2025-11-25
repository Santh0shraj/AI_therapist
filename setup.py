import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def create_directories():
    """Create necessary directories"""
    directories = ['models', 'data', 'screenshots']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def check_camera():
    """Check if camera is accessible"""
    import cv2
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("✓ Camera is accessible")
        cap.release()
        return True
    else:
        print("✗ Camera not accessible")
        return False

if __name__ == "__main__":
    print("Setting up Emotion Detection System...")
    create_directories()
    
    try:
        install_requirements()
        print("✓ Requirements installed successfully")
    except Exception as e:
        print(f"✗ Error installing requirements: {e}")
    
    check_camera()
    print("\nSetup complete! Next steps:")
    print("1. Organize your FER-2013 data in the correct folder structure")
    print("2. Run: python train_emotion_model.py")
    print("3. Run: python realtime_emotion_detection.py")