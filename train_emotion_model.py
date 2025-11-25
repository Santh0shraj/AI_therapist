import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class EmotionTrainer:
    def __init__(self, data_path='data/FER-2013'):
        self.data_path = data_path
        self.img_size = 48
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.model = None
        
    def load_data(self):
        print("Loading training data...")
        X = []
        y = []
        
        for emotion_idx, emotion in enumerate(self.emotions):
            emotion_path = os.path.join(self.data_path, 'train', emotion)
            if not os.path.exists(emotion_path):
                print(f"Warning: {emotion_path} not found, skipping...")
                continue
                
            for img_file in os.listdir(emotion_path)[:1000]:  # Limit for memory
                if img_file.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(emotion_path, img_file)
                    try:
                        # Read and preprocess image
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = cv2.resize(img, (self.img_size, self.img_size))
                            img = img.astype('float32') / 255.0
                            X.append(img)
                            y.append(emotion_idx)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
        
        if not X:
            raise Exception("No training data found! Check your data path.")
            
        X = np.array(X).reshape(-1, self.img_size, self.img_size, 1)
        y = keras.utils.to_categorical(y, len(self.emotions))
        
        print(f"Loaded {len(X)} images")
        return X, y
    
    def create_model(self):
        model = keras.Sequential([
            # First Conv Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_size, self.img_size, 1)),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Second Conv Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Third Conv Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Fourth Conv Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(len(self.emotions), activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, epochs=50):
        X, y = self.load_data()
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = self.create_model()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
            keras.callbacks.ModelCheckpoint('models/emotion_model.h5', save_best_only=True)
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model
        self.model.save('models/emotion_model.h5')
        
        # Save model architecture
        model_json = self.model.to_json()
        with open("models/emotion_model.json", "w") as json_file:
            json_file.write(model_json)
            
        # Plot training history
        self.plot_training_history(history)
        
        return history
    
    def plot_training_history(self, history):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('models/training_history.png')
        plt.show()
    def test_interactive_model():
        """Quick test to verify the model works for interactive sessions"""
        model_path = 'models/emotion_model.h5'
        
        if not os.path.exists(model_path):
            print("❌ Model not found! Please train the model first.")
            return
        
        print("🧪 Testing model for interactive session...")
        model = keras.models.load_model(model_path)
        
        # Test with sample data
        test_input = np.random.random((1, 48, 48, 1)).astype('float32')
        prediction = model.predict(test_input, verbose=0)
        
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        predicted_emotion = emotions[np.argmax(prediction[0])]
        confidence = np.max(prediction[0])
        
        print(f"✅ Model ready for interactive sessions!")
        print(f"Sample prediction: {predicted_emotion} (confidence: {confidence:.2f})")
        
        # Test response generation
        therapist = InteractiveEmotionTherapist(model_path)
        test_response = therapist.generate_dynamic_response('happy', 0.8, True)
        print(f"Sample interaction: {test_response}")
if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    trainer = EmotionTrainer()
    print("Starting training...")
    trainer.train(epochs=50)