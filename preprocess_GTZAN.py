# import what we need and set up globals

import os
import librosa
import math
import json

DATASET_PATH = "genres_original"
JSON_PATH = "data\GTZAN_dict.json"

SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def read_wavs(dataset_path, json_path):
    
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
          # dictionary to store data
        data = {
            "mapping": [],
            "mfcc": [],
            "labels": []
        }  
        # ensure we are not at the root level
        if dirpath is not dataset_path:

            # save semantic label
            dirpath_components = dirpath.split("\\")
            semantic_label = dirpath_components[-1]
            print("\n semantic label for dir is {}\n\n\n".format(semantic_label))
            
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                print("file is {}".format(file_path))
                if 'wav' in file_path:
                    data["mapping"].append(semantic_label)

        with open(json_path, "w") as fp:
            json.dump(data, fp, indent=4)

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    # dictionary to store data
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }
    
    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)
    
    # loop through all the genres
    
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # ensure we are not at the root level
        if dirpath is not dataset_path:
            
            # save semantic label
            dirpath_components = dirpath.split("\\")
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing {}".format(semantic_label))
        
            # process files for a specific genre
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                
                # process segments extracting mfcc and store data
                
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment
                    
                    
                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                                sr=sr, n_fft=n_fft, n_mfcc=n_mfcc,
                                                hop_length= hop_length)
                    
                    mfcc = mfcc.T
                    
                # store mfcc for segment if it has expected length
                if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                    data["mfcc"].append(mfcc.tolist())
                    data["labels"].append(i-1)
                    
                    print("{}, segment:{}".format(file_path, s))
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)