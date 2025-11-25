from realtime_emotion_detection import InteractiveEmotionTherapist
import os

def quick_interaction_test():
    """Test the interaction system without camera"""
    print("🧪 Quick Interaction Test")
    print("=" * 40)
    
    # Test response generation for different emotions
    test_cases = [
        ('happy', 0.9, True),
        ('sad', 0.7, False),
        ('angry', 0.8, True),
        ('neutral', 0.6, False)
    ]
    
    therapist = InteractiveEmotionTherapist()
    
    for emotion, confidence, changed in test_cases:
        response = therapist.generate_dynamic_response(emotion, confidence, changed)
        print(f"Emotion: {emotion} (conf: {confidence}, changed: {changed})")
        print(f"Response: {response}")
        print("-" * 40)

if __name__ == "__main__":
    if os.path.exists('models/emotion_model.h5'):
        quick_interaction_test()
    else:
        print("Please train the model first using train_emotion_model.py")