# Testing

import keras
import argparse
import json
import statistics
import librosa
import numpy as np
import math

def preprocessing(music_path,n_mfcc=20,hop_length=512,n_fft=2048,num_of_segments=5,SAMPELING_RATE=22050):
    
    mfccs = []
    signal,sr = librosa.core.load(music_path,sr=SAMPELING_RATE)
    DURATION = librosa.core.get_duration(signal,sr=SAMPELING_RATE,n_fft=n_fft,hop_length=hop_length)
    NUM_SAMPLES = SAMPELING_RATE*DURATION
    NUM_SAMPLES_PER_SEGMENT = int(NUM_SAMPLES/num_of_segments)
    num_mfcc_vectors_per_segment = math.ceil(NUM_SAMPLES_PER_SEGMENT / hop_length)
    
    segment = 0

    while 1:
      try:
        start_segment = segment*132358 #NUM_SAMPLES_PER_SEGMENT*segment
        end_segment = start_segment+132358 #start_segment+NUM_SAMPLES_PER_SEGMENT

        mfcc = librosa.feature.mfcc(signal[start_segment:end_segment],sr=SAMPELING_RATE , n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        if mfcc.shape==(20,259):
          mfccs.append(mfcc)

        segment+=1
      
      except:
        break

    

    return np.array(mfccs)

def index_2_label(pred):
    global mapping

    return(mapping[pred])

    
def predict(model,mfccs):
    pred = []
    
    for mfcc in mfccs:
        prediction = model.predict(mfcc.reshape(-1,20,259,1))
        pred.append(np.argmax(prediction))

    final_pred = statistics.mode(pred)

    label = index_2_label(final_pred)

    return(label)
        
model = keras.models.load_model("model.h5")

with open("data.json","r") as json_file:
  mapping = np.array(json.load(json_file)["mapping"])


parser = argparse.ArgumentParser()

parser.add_argument('-path', action='store', type=str,
                    help='Path of the music')

results = parser.parse_args()

mfccs = preprocessing(results.path)

prediction = predict(model,mfccs)

print("The genre of the song is: {}".format(prediction))
