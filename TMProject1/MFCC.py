from python_speech_features import mfcc
import glob
from scipy.io import wavfile
import numpy as np

#reading wave files from path and adding to list
file_list = glob.glob("D:/Inne Projekty z Programowania/TMProject1/train/*.wav")
only_name = []
for filename in file_list:
    temp = filename.split("\\")
    only_name.append(temp[1])
print(only_name)

mfcc_file_dict = {}
#iterating through list
for wave_name in only_name:
    fs, data = wavfile.read('D:/Inne Projekty z Programowania/TMProject1/train/'+wave_name)

    #creating MFCC matrix
    mfcc_matrix = mfcc(signal=data, samplerate=fs)
    mfcc_file_dict[wave_name] = mfcc_matrix

    

for key in mfcc_file_dict.keys():
    temp_name = key.split('.') #splits srting on file_name and wav
    text_name = temp_name[0]+".txt"

    file = open(text_name, 'w')
    for row in mfcc_file_dict[key]:
        file.write(str(row))
    file.close()