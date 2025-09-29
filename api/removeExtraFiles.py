import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

start_folder = os.getcwd() + '/patient-vocal-dataset'
class_names = [i for i in os.listdir(start_folder) if not i.startswith('.') ]

for i in class_names:
    class_path = os.path.join(start_folder, i)

    wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
    print(f"Number of wav files in {i}: {len(wav_files)}")

    egg_wav_files = [f for f in wav_files if f.endswith('egg.wav')]
    print(f"Number of egg wav files in {i}: {len(egg_wav_files)}")
    print(f"Number of files after removal: {len(wav_files) - len(egg_wav_files)}")
    if len(egg_wav_files) == 0: print("No files to remove, skipping delete operation...")
    else:
        # removing only egg wav files, keeping wav files intact
        for f in egg_wav_files:
            os.remove(os.path.join(class_path, f))
        print("Removed all egg wav files from class:", i)
    print()