from python_speech_features import mfcc
import glob
from scipy.io import wavfile
import numpy as np
import pickle
def mfcc_to_pickle():
    #reading wave files from path and adding to list
    file_list = glob.glob("D:/Inne Projekty z Programowania/TMProject1/train/*.wav")
    only_name = []
    for filename in file_list:
        temp = filename.split("\\")
        only_name.append(temp[1])

    #iterating through list
    mfcc_file_dict = {}
    for wave_name in only_name:
        fs, data = wavfile.read("D:/Inne Projekty z Programowania/TMProject1/train/"+wave_name)

        #creating MFCC matrix
        temp_name = wave_name.split('.')#cutting off extension
        mfcc_matrix = mfcc(signal=data, samplerate=fs)
        mfcc_file_dict[temp_name[0]] = mfcc_matrix


    #serialisation and saving

    pickle_out = open('train.pickle', 'wb')
    pickle.dump(mfcc_file_dict, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
    pickle_out.close()

def load_mfcc_from_pickle():
    #loading training data
    pickle_in = open('train.pickle', 'rb')
    mfcc_file_dict = pickle.load(pickle_in)
    #print(mfcc_file_dict['AO1M1_0_'])

    #Spliting on smaller boards

    #key=number value=list of mfcc
    samples_dict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}

    for key in mfcc_file_dict.keys():
        number = key.split('_')[1]
        samples_dict[int(number)].append(mfcc_file_dict[key])
    return samples_dict

#mfcc_to_pickle()

main_dict=load_mfcc_from_pickle()

print(np.shape(main_dict[1]))
