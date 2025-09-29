import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
import base64
import random
import tempfile

# Load model
model = tf.keras.models.load_model('92cnn.keras')

# Dataset
start_folder = 'patient-vocal-dataset'
class_names = [i for i in os.listdir(start_folder) if not i.startswith('.')]
test_folder = start_folder

if not os.path.exists(test_folder):
    os.makedirs(test_folder, exist_ok=True)
    for class_name in class_names:
        class_test_folder = os.path.join(test_folder, class_name)
        os.makedirs(class_test_folder, exist_ok=True)


def audio_to_spectrogram_array(audio_data, sample_rate):
    spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=128)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    plt.figure(figsize=(10, 5))
    librosa.display.specshow(log_spectrogram, sr=sample_rate, x_axis='time', y_axis='mel')
    plt.axis('off')

    fig = plt.gcf()
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    img_array = tf.image.resize(img_array, (256, 512))
    img_array = tf.cast(img_array, tf.uint8)
    return img_array


def create_spectrogram_image(audio_data, sample_rate, title=None):
    plt.figure(figsize=(10, 5))
    spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=128)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    librosa.display.specshow(log_spectrogram, sr=sample_rate, x_axis='time', y_axis='mel')
    if title:
        plt.title(title)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str


async def predict_audio(file):
    audio_bytes = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as input_file:
        input_path = input_file.name
        input_file.write(audio_bytes)

    try:
        audio_data, sample_rate = librosa.load(input_path, sr=None)

        img_array = audio_to_spectrogram_array(audio_data, sample_rate)
        img_array = tf.cast(img_array, tf.float32) / 255.0
        img_array = tf.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class_idx = np.argmax(prediction[0])
        predicted_class = class_names[predicted_class_idx]
        confidence = float(prediction[0][predicted_class_idx])

        img_str = create_spectrogram_image(audio_data, sample_rate, f"Prediction: {predicted_class}")

        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "spectrogram": img_str
        }
    finally:
        if os.path.exists(input_path):
            os.unlink(input_path)


def get_random_samples():
    samples = []
    classes = os.listdir(test_folder)

    for _ in range(3):
        random_class = random.choice(classes)
        class_path = os.path.join(test_folder, random_class)

        wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
        if not wav_files:
            continue

        random_file = random.choice(wav_files)
        file_path = os.path.join(class_path, random_file)

        audio_data, sample_rate = librosa.load(file_path, sr=None)
        img_array = audio_to_spectrogram_array(audio_data, sample_rate)

        img_array_normalized = tf.cast(img_array, tf.float32) / 255.0
        img_array_normalized = tf.expand_dims(img_array_normalized, axis=0)

        prediction = model.predict(img_array_normalized)
        predicted_class_idx = np.argmax(prediction[0])
        predicted_class = class_names[predicted_class_idx]
        confidence = float(prediction[0][predicted_class_idx])

        img_str = create_spectrogram_image(audio_data, sample_rate)

        with open(file_path, "rb") as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')

        samples.append({
            "audio_path": file_path,
            "true_class": random_class,
            "prediction": predicted_class,
            "confidence": confidence,
            "spectrogram": img_str,
            "audio_base64": audio_base64
        })

    return samples

if __name__=='__main__':
    test = get_random_samples()
    print(len(test))