import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import time
import requests
import json
import threading
import random

class RealTimeEmotionDetection:
    def __init__(self, model_path='models/emotion_model.h5'):
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # Initialize face detector
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except:
            # Alternative path for face detector
            self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
        # Load model
        if os.path.exists(model_path):
            print("Loading model...")
            self.model = keras.models.load_model(model_path)
            print("Model loaded successfully!")
        else:
            raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
        
        # Interactive session state
        self.conversation_active = False
        self.current_emotion = "Neutral"
        self.emotion_confidence = 0.0
        self.last_emotion_time = time.time()
        self.emotion_history = []
        self.conversation_history = []
        
        # API configuration
        self.api_endpoints = {
            'openai': 'https://api.openai.com/v1/chat/completions',
            'huggingface': 'https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium',
            'local_llm': 'http://localhost:5001/api/chat'  # For local models like Ollama
        }
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        print("Interactive Emotion Therapy Session Ready!")
        print("Available APIs: OpenAI, HuggingFace, Local LLM")
    
    def setup_api_client(self, api_type='openai', api_key=None):
        """Setup API client for interactive conversations"""
        self.api_type = api_type
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        if api_type == 'openai' and not self.api_key:
            print("Warning: No API key provided. Using simulated responses.")
        
        print(f"API Client configured for: {api_type}")
    
    def generate_ai_response(self, user_input=None, current_emotion=None, confidence=None):
        """Generate AI response using configured API"""
        
        # If no API key, use simulated responses
        if not hasattr(self, 'api_key') or not self.api_key:
            return self._generate_simulated_response(user_input, current_emotion, confidence)
        
        try:
            if self.api_type == 'openai':
                return self._call_openai_api(user_input, current_emotion, confidence)
            elif self.api_type == 'huggingface':
                return self._call_huggingface_api(user_input, current_emotion, confidence)
            elif self.api_type == 'local_llm':
                return self._call_local_llm(user_input, current_emotion, confidence)
            else:
                return self._generate_simulated_response(user_input, current_emotion, confidence)
        except Exception as e:
            print(f"API Error: {e}")
            return self._generate_simulated_response(user_input, current_emotion, confidence)
    
    def _call_openai_api(self, user_input, current_emotion, confidence):
        """Call OpenAI GPT API for responses"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        # Build conversation context
        messages = [
            {
                "role": "system",
                "content": """You are a compassionate therapeutic AI assistant. You respond to users based on their facial emotions detected in real-time. 
                Be empathetic, supportive, and engaging. Acknowledge their current emotion and respond appropriately.
                Keep responses concise (1-2 sentences) for real-time interaction."""
            }
        ]
        
        # Add conversation history
        for entry in self.conversation_history[-6:]:  # Last 6 exchanges
            if entry['user']:
                messages.append({"role": "user", "content": entry['user']})
            if entry['ai']:
                messages.append({"role": "assistant", "content": entry['ai']})
        
        # Add current context
        emotion_context = f"The user currently appears {current_emotion.lower()} (confidence: {confidence:.2f})."
        if user_input:
            prompt = f"{emotion_context} User says: {user_input}"
        else:
            prompt = f"{emotion_context} Respond to their current emotional state."
        
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": "gpt-3.5-turbo",
            "messages": messages,
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        response = requests.post(self.api_endpoints['openai'], headers=headers, json=data, timeout=10)
        response.raise_for_status()
        
        ai_response = response.json()['choices'][0]['message']['content'].strip()
        return ai_response
    
    def _call_huggingface_api(self, user_input, current_emotion, confidence):
        """Call HuggingFace API for responses"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        prompt = f"The user appears {current_emotion.lower()}. "
        if user_input:
            prompt += f"They said: {user_input}. "
        prompt += "Provide a supportive therapeutic response."
        
        data = {"inputs": prompt}
        
        response = requests.post(self.api_endpoints['huggingface'], headers=headers, json=data, timeout=15)
        response.raise_for_status()
        
        ai_response = response.json()[0]['generated_text'].strip()
        return ai_response
    
    def _call_local_llm(self, user_input, current_emotion, confidence):
        """Call local LLM API (like Ollama)"""
        prompt = f"User emotion: {current_emotion} (confidence: {confidence:.2f}). "
        if user_input:
            prompt += f"User input: {user_input}. "
        prompt += "Provide a brief, empathetic therapeutic response."
        
        data = {
            "model": "llama2",  # or your local model name
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(self.api_endpoints['local_llm'], json=data, timeout=30)
        response.raise_for_status()
        
        ai_response = response.json()['response'].strip()
        return ai_response
    
    def _generate_simulated_response(self, user_input, current_emotion, confidence):
        """Generate simulated responses when no API is available"""
        
        emotion_responses = {
            'Happy': [
                "I see that smile! What's bringing you joy right now?",
                "Your positive energy is wonderful! Want to share what's making you happy?",
                "That's a genuine smile! How are you feeling in this moment?"
            ],
            'Sad': [
                "I notice you seem a bit down. Would you like to talk about what's on your mind?",
                "You look like you might be having a tough moment. I'm here to listen.",
                "I sense some heaviness. Remember, it's okay to share what you're feeling."
            ],
            'Angry': [
                "I can see some tension. Would it help to talk about what's bothering you?",
                "You seem frustrated. Sometimes putting feelings into words can help.",
                "I notice some anger. Let's work through this together."
            ],
            'Fear': [
                "You seem a bit anxious. You're in a safe space here.",
                "I sense some nervousness. Would you like to talk about what's worrying you?",
                "You look concerned. Remember to breathe - I'm here with you."
            ],
            'Surprise': [
                "Something seems to have caught your attention!",
                "You look surprised! What's happening?",
                "That's an unexpected expression! Care to share what just happened?"
            ],
            'Disgust': [
                "You seem really bothered by something.",
                "I notice strong disapproval. What's triggering this reaction?",
                "You look like something doesn't sit right with you."
            ],
            'Neutral': [
                "How are you feeling in this moment?",
                "You seem contemplative. What's on your mind?",
                "Hello! I'm here when you're ready to talk."
            ]
        }
        
        if user_input:
            # Respond to user input
            input_lower = user_input.lower()
            if any(word in input_lower for word in ['sad', 'unhappy', 'depressed']):
                return "I hear that you're feeling down. Would you like to talk more about what's causing these feelings?"
            elif any(word in input_lower for word in ['happy', 'good', 'great']):
                return "I'm glad you're feeling positive! What's contributing to these good feelings?"
            elif any(word in input_lower for word in ['angry', 'mad', 'frustrated']):
                return "I understand you're feeling upset. What's been frustrating you?"
            else:
                return f"Thank you for sharing. Given that you appear {current_emotion.lower()}, how does that relate to what you just said?"
        else:
            # Respond to emotion only
            return random.choice(emotion_responses.get(current_emotion, emotion_responses['Neutral']))
    
    def start_conversation(self):
        """Start an interactive conversation session"""
        self.conversation_active = True
        self.conversation_history = []
        
        opening_lines = [
            "Hello! I can see you're here. How are you feeling today?",
            "Welcome to our session! I'm here to listen and support you.",
            "Hi there! I notice you've started. Feel free to express yourself."
        ]
        
        opening = random.choice(opening_lines)
        print(f"🤖 Therapist: {opening}")
        
        # Log the opening
        self.conversation_history.append({
            'timestamp': time.time(),
            'user': None,
            'ai': opening,
            'emotion': self.current_emotion,
            'confidence': self.emotion_confidence
        })
    
    def process_user_input(self, user_input):
        """Process user input and generate AI response"""
        if not self.conversation_active:
            self.start_conversation()
            return
        
        # Generate AI response
        ai_response = self.generate_ai_response(
            user_input=user_input,
            current_emotion=self.current_emotion,
            confidence=self.emotion_confidence
        )
        
        print(f"👤 You: {user_input}")
        print(f"🤖 Therapist: {ai_response}")
        
        # Update conversation history
        self.conversation_history.append({
            'timestamp': time.time(),
            'user': user_input,
            'ai': ai_response,
            'emotion': self.current_emotion,
            'confidence': self.emotion_confidence
        })
        
        # Keep history manageable
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    def auto_respond_to_emotion(self):
        """Automatically respond to significant emotion changes"""
        if not self.conversation_active:
            return
        
        current_time = time.time()
        time_since_last = current_time - self.last_emotion_time
        
        # Only respond if significant time has passed and emotion is clear
        if time_since_last > 10 and self.emotion_confidence > 0.7:
            ai_response = self.generate_ai_response(
                current_emotion=self.current_emotion,
                confidence=self.emotion_confidence
            )
            
            print(f"🤖 Therapist: {ai_response}")
            
            # Update conversation history
            self.conversation_history.append({
                'timestamp': time.time(),
                'user': None,
                'ai': ai_response,
                'emotion': self.current_emotion,
                'confidence': self.emotion_confidence
            })
            
            self.last_emotion_time = current_time
    
    def preprocess_face(self, face_roi):
        """Preprocess face ROI for emotion prediction"""
        # Convert to grayscale if needed
        if len(face_roi.shape) == 3:
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize to model input size
        face_roi = cv2.resize(face_roi, (48, 48))
        
        # Normalize
        face_roi = face_roi.astype('float32') / 255.0
        
        # Reshape for model
        face_roi = face_roi.reshape(1, 48, 48, 1)
        
        return face_roi
    
    def predict_emotion(self, face_roi):
        """Predict emotion from face ROI"""
        processed_face = self.preprocess_face(face_roi)
        predictions = self.model.predict(processed_face, verbose=0)
        emotion_idx = np.argmax(predictions[0])
        confidence = predictions[0][emotion_idx]
        
        return self.emotions[emotion_idx], confidence, predictions[0]
    
    def draw_emotion_info(self, frame, face_rect, emotion, confidence, all_predictions):
        """Draw emotion information on the frame"""
        x, y, w, h = face_rect
        
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw emotion text
        emotion_text = f"{emotion}: {confidence:.2f}"
        cv2.putText(frame, emotion_text, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw conversation status
        status = "Conversation: ACTIVE" if self.conversation_active else "Conversation: INACTIVE"
        cv2.putText(frame, status, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if self.conversation_active else (255, 255, 255), 2)
        
        # Draw confidence bar
        bar_width = 150
        bar_height = 15
        confidence_width = int(confidence * bar_width)
        
        cv2.rectangle(frame, (x, y+h+5), (x+bar_width, y+h+5+bar_height), (255, 255, 255), 1)
        cv2.rectangle(frame, (x, y+h+5), (x+confidence_width, y+h+5+bar_height), (0, 255, 0), -1)
    
    def update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1
        if self.frame_count >= 30:
            end_time = time.time()
            self.fps = self.frame_count / (end_time - self.start_time)
            self.frame_count = 0
            self.start_time = end_time
    
    def run(self):
        """Run real-time emotion detection with interactive API"""
        cap = cv2.VideoCapture(0)
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("🤖 Starting Real-Time Emotion Detection with Interactive API")
        print("=" * 60)
        print("Controls:")
        print("  'q' - Quit")
        print("  'c' - Start/Stop conversation")
        print("  't' - Type a message")
        print("  's' - Save screenshot")
        print("  'a' - Auto-respond to emotions")
        print("=" * 60)
        
        auto_respond = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Update FPS
            self.update_fps()
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(50, 50)
            )
            
            # Process each face
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                
                try:
                    emotion, confidence, all_predictions = self.predict_emotion(face_roi)
                    
                    # Update current emotion state
                    emotion_changed = (emotion != self.current_emotion)
                    self.current_emotion = emotion
                    self.emotion_confidence = confidence
                    
                    self.draw_emotion_info(frame, (x, y, w, h), emotion, confidence, all_predictions)
                    
                    # Auto-respond to significant emotion changes
                    if auto_respond and emotion_changed and confidence > 0.7:
                        self.auto_respond_to_emotion()
                        
                except Exception as e:
                    print(f"Error predicting emotion: {e}")
                    continue
            
            # Display FPS and controls
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Emotion: {self.current_emotion}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, "Press 'c' to start conversation, 't' to type", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('Real-Time Emotion Detection + Interactive API', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Toggle conversation
                if self.conversation_active:
                    self.conversation_active = False
                    print("Conversation ended.")
                else:
                    self.start_conversation()
            elif key == ord('t'):
                # Type a message
                user_input = input("\n💬 Your message: ").strip()
                if user_input:
                    self.process_user_input(user_input)
            elif key == ord('s'):
                # Save screenshot
                timestamp = int(time.time())
                cv2.imwrite(f'screenshot_{timestamp}.png', frame)
                print(f"Screenshot saved: screenshot_{timestamp}.png")
            elif key == ord('a'):
                # Toggle auto-respond
                auto_respond = not auto_respond
                status = "ENABLED" if auto_respond else "DISABLED"
                print(f"Auto-respond: {status}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Session summary
        if self.conversation_history:
            print(f"\nSession summary: {len(self.conversation_history)} interactions")
            print("Thank you for the conversation! 👋")

if __name__ == "__main__":
    try:
        detector = RealTimeEmotionDetection()
        
        # Configure API (choose one)
        print("\nSelect API configuration:")
        print("1. OpenAI GPT (requires API key)")
        print("2. HuggingFace (requires API key)") 
        print("3. Local LLM (requires local server)")
        print("4. Simulated responses (no API needed)")
        
        choice = input("Enter choice (1-4, default 4): ").strip()
        
        if choice == '1':
            api_key = input("Enter OpenAI API key (or press Enter to use env var): ").strip()
            detector.setup_api_client('openai', api_key or None)
        elif choice == '2':
            api_key = input("Enter HuggingFace API key: ").strip()
            detector.setup_api_client('huggingface', api_key)
        elif choice == '3':
            detector.setup_api_client('local_llm')
        else:
            detector.setup_api_client('simulated')
        
        detector.run()
        
    except FileNotFoundError as e:
        print(e)
        print("Please run train_emotion_model.py first to train the model.")
    except Exception as e:
        print(f"Error: {e}")