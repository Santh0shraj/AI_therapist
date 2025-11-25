import cv2
import numpy as np
import os

def organize_fer2013_dataset(csv_path, output_dir):
    """
    If your FER-2013 data is in CSV format, use this function to organize it into folders
    """
    import pandas as pd
    
    # Create output directories
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    for emotion in emotions:
        os.makedirs(os.path.join(output_dir, 'train', emotion), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'test', emotion), exist_ok=True)
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    for index, row in df.iterrows():
        try:
            emotion = emotions[row['emotion']]
            usage = row['Usage']
            pixels = np.array(row['pixels'].split(), dtype='float32').reshape(48, 48)
            
            # Determine save path
            if usage == 'Training':
                save_dir = os.path.join(output_dir, 'train', emotion)
            else:
                save_dir = os.path.join(output_dir, 'test', emotion)
            
            # Save image
            img_path = os.path.join(save_dir, f'{index}.png')
            cv2.imwrite(img_path, pixels)
            
        except Exception as e:
            print(f"Error processing row {index}: {e}")
    
    print("Dataset organization completed!")

def test_camera():
    """Test if camera is working"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Camera not accessible")
        return False
    
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        print("Camera test successful!")
        return True
    else:
        print("Error: Could not read from camera")
        return False