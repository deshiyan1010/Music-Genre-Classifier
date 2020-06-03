#Data Preprocessing

import librosa
import json
import math
import numpy as np
import os
import argparse

def create_dataset(n_mfcc=20,hop_length=512,n_fft=2048,num_of_segments=5,path='/content/Music-Genre-Classifier/genres'): 
  
  
  DIR_PATH = path                              

  SAMPELING_RATE = 22050
  data = {
      "mapping":[],
      "MFCCs":[],
      "label":[]
  }

  folder_list = []
  for files in os.listdir(DIR_PATH):
    if "."not in str(files):
      folder_list.append(files)

  for folder in folder_list:
    FOLDER_PATH = os.path.join(DIR_PATH,folder)
    
    for files in os.listdir(FOLDER_PATH):

      if folder not in data["mapping"]:
        data["mapping"].append(folder)
      file_path = os.path.join(DIR_PATH,folder,files)
      signal,sr = librosa.core.load(file_path,sr=SAMPELING_RATE)
      DURATION = librosa.core.get_duration(signal,sr=SAMPELING_RATE,n_fft=n_fft,hop_length=hop_length)
      NUM_SAMPLES = SAMPELING_RATE*DURATION
      NUM_SAMPLES_PER_SEGMENT = int(NUM_SAMPLES/num_of_segments)
      num_mfcc_vectors_per_segment = math.ceil(NUM_SAMPLES_PER_SEGMENT / hop_length)
      for segment in range (num_of_segments):
        start_segment = NUM_SAMPLES_PER_SEGMENT*segment
        end_segment = start_segment+NUM_SAMPLES_PER_SEGMENT

        mfcc = librosa.feature.mfcc(signal[start_segment:end_segment],sr=SAMPELING_RATE , n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        #print(len(mfcc))
        if np.array(mfcc).shape == (20,259):
          data["MFCCs"].append(mfcc.tolist())
          data["label"].append(data["mapping"].index(folder))
          print("{}, segment:{}".format(file_path, segment+1))
      break 
  save_path = os.path.join(DIR_PATH,"..","data.json")
  with open(save_path, "w") as fp:
    json.dump(data, fp, indent=4)
  return data

parser = argparse.ArgumentParser()

parser.add_argument('-path', action='store', type=str,
                    help='Path of the music folder')

results = parser.parse_args()


data = create_dataset(path=results.path)
