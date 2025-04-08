from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import librosa
import matplotlib.pyplot as plt
import io
import base64
import os
import random
from pydantic import BaseModel
import uvicorn

# pip install fastapi uvicorn librosa tensorflow matplotlib pydantic numpy
# install ffmpeg
# python -m uvicorn main:app

app = FastAPI(title="Voice Disease Detection API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the pre-trained model
model = tf.keras.models.load_model('92cnn.keras')

# Define class names (modify these based on your actual classes)
start_folder = 'patient-vocal-dataset'
class_names = os.listdir(start_folder)

# Create test set folder if it doesn't exist
test_folder = 'test-set'
if not os.path.exists(test_folder):
    os.makedirs(test_folder)
    # Create subdirectories for each class
    for class_name in class_names:
        class_test_folder = os.path.join(test_folder, class_name)
        if not os.path.exists(class_test_folder):
            os.makedirs(class_test_folder)
        
        # Copy 20% of each class's files to test set
        class_folder = os.path.join(start_folder, class_name)
        files = [f for f in os.listdir(class_folder) if f.endswith('.wav')]
        test_files = random.sample(files, min(len(files) // 5, 20))
        
        for file in test_files:
            source = os.path.join(class_folder, file)
            destination = os.path.join(class_test_folder, file)
            # Copy file (symbolic link or actual copy)
            try:
                os.symlink(os.path.abspath(source), destination)
            except:
                import shutil
                shutil.copy2(source, destination)

# Function to convert audio to spectrogram array
def audio_to_spectrogram_array(audio_data, sample_rate):
    try:
        # Ensure audio data is not empty
        if len(audio_data) == 0:
            raise ValueError("Audio data is empty")
            
        # Create spectrogram
        spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=128)
        log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        
        # Convert to image format
        plt.figure(figsize=(10, 5))
        librosa.display.specshow(log_spectrogram, sr=sample_rate, x_axis='time', y_axis='mel')
        plt.axis('off')
        
        # Save figure to buffer
        fig = plt.gcf()
        fig.canvas.draw()
        # Convert to numpy array
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        
        # Resize to target dimensions
        img_array = tf.image.resize(img_array, (256, 512))
        img_array = tf.cast(img_array, tf.uint8)
        
        return img_array
    except Exception as e:
        plt.close()  # Make sure to close any open plots
        raise e

# Function to create spectrogram image and return base64
def create_spectrogram_image(audio_data, sample_rate, title=None):
    try:
        plt.figure(figsize=(10, 5))
        spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=128)
        log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        librosa.display.specshow(log_spectrogram, sr=sample_rate, x_axis='time', y_axis='mel')
        if title:
            plt.title(title)
        
        # Save the spectrogram to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        # Convert to base64 for sending to frontend
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return img_str
    except Exception as e:
        plt.close()  # Make sure to close any open plots
        raise e

@app.post("/predict/")
async def predict_audio(file: UploadFile = File(...)):
    # Read the audio file
    audio_bytes = await file.read()
    
    try:
        import tempfile
        import os
        import subprocess
        import shutil
        
        # Create temporary files for input and output
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as input_file:
            input_path = input_file.name
            input_file.write(audio_bytes)
        
        # Create output file path with .wav extension
        output_path = os.path.splitext(input_path)[0] + "_converted.wav"
        
        try:
            # Use FFmpeg to convert to WAV format that librosa can handle
            ffmpeg_path = shutil.which("ffmpeg")
            if not ffmpeg_path:
                ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"  # Fallback to known location

            subprocess.run(
                [ffmpeg_path, "-i", input_path, "-ar", "44100", "-ac", "1", output_path], 
                check=True, 
                capture_output=True
            )
            
            # Now load with librosa from the converted file
            audio_data, sample_rate = librosa.load(output_path, sr=None)
            
            # Continue with existing processing
            img_array = audio_to_spectrogram_array(audio_data, sample_rate)
            
            # Normalize and prepare for model
            img_array = tf.cast(img_array, tf.float32) / 255.0
            img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension
            
            # Make prediction
            prediction = model.predict(img_array)
            predicted_class_idx = np.argmax(prediction[0])
            predicted_class = class_names[predicted_class_idx]
            confidence = float(prediction[0][predicted_class_idx])
            
            # Create spectrogram image to send to frontend
            img_str = create_spectrogram_image(audio_data, sample_rate, f"Prediction: {predicted_class}")
            
            return {
                "prediction": predicted_class,
                "confidence": confidence,
                "spectrogram": img_str
            }
            
        finally:
            # Clean up temporary files
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
                
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"FFmpeg conversion error: {e.stderr.decode()}")
    except Exception as e:
        import traceback
        stack_trace = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}\n{stack_trace}")

@app.get("/random-samples/")
async def get_random_samples():
    # Use the test set folder
    folder = test_folder
    
    if not os.path.exists(folder):
        raise HTTPException(status_code=404, detail=f"Test dataset folder not found: {folder}")
    
    samples = []
    try:
        # Get all class folders
        classes = os.listdir(folder)
        for _ in range(3):  # Get 3 random samples
            # Choose a random class
            random_class = random.choice(classes)
            class_path = os.path.join(folder, random_class)
            
            # Get all WAV files in that class
            wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
            if not wav_files:
                continue
                
            # Choose a random WAV file
            random_file = random.choice(wav_files)
            file_path = os.path.join(class_path, random_file)
            
            # Process the audio file
            audio_data, sample_rate = librosa.load(file_path, sr=None)
            
            # Convert to spectrogram for prediction
            img_array = audio_to_spectrogram_array(audio_data, sample_rate)
            
            # Normalize and prepare for model
            img_array_normalized = tf.cast(img_array, tf.float32) / 255.0
            img_array_normalized = tf.expand_dims(img_array_normalized, axis=0)  # Add batch dimension
            
            # Make prediction
            prediction = model.predict(img_array_normalized)
            predicted_class_idx = np.argmax(prediction[0])
            predicted_class = class_names[predicted_class_idx]
            confidence = float(prediction[0][predicted_class_idx])
            
            # Create spectrogram image
            img_str = create_spectrogram_image(audio_data, sample_rate)
            
            # Read the audio file as base64
            with open(file_path, "rb") as audio_file:
                audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
            
            # Add sample info to samples list
            samples.append({
                "audio_path": file_path,
                "true_class": random_class,
                "prediction": predicted_class,
                "confidence": confidence,
                "spectrogram": img_str,
                "audio_base64": audio_base64
            })
            
    except Exception as e:
        import traceback
        stack_trace = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Error getting random samples: {str(e)}\n{stack_trace}")
    
    return samples

@app.get("/")
async def root():
    return {"message": "Voice Disease Detection API is running"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)