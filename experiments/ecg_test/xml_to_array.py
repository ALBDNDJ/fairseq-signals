import pandas as pd
import numpy as np
import xmltodict
import base64
import struct
import argparse
import os
import sys

def file_path(path):
    filepath = path
    for dirName, subdirList, fileList in os.walk(filepath):
        for filename in fileList:
            if ".xml" in filename.lower():
                ekg_file_list.append(os.path.join(dirName, filename))

if not os.path.exists(os.getcwd() + '/waveforms_output/'):
    os.mkdir(os.getcwd() + '/waveforms_output/')




def xml_to_np_array_file(path_to_xml, path_to_output = os.getcwd()):

    with open(path_to_xml, 'rb') as fd:
        dic = xmltodict.parse(fd.read().decode('utf8'))

    try:
        pt_id = dic['GTRestingECGData']['Patient']['PatientID']
    except:
        print("no PatientID")
        pt_id = "none"

    try:
        AcquisitionDateTime = dic['GTRestingECGData']['Test']['TestDateTime']
    except:
        print("no TestDateTime")
        AcquisitionDateTime = "none"

    #need to instantiate leads in the proper order for the model
    lead_order = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    lead_data =  dict.fromkeys(lead_order)
    lead = dic['GTRestingECGData']['RhythmWaveform']['LeadData']
    for leadid in range(len(lead)):
        sample_length = len(np.array(lead[leadid]['WaveFormData'].split()))
        if sample_length == 5000:
            lead_data[lead[leadid]['LeadID']] = np.array(lead[leadid]['WaveFormData'].split()).astype(int)
        else:
            print('ensures all leads have 5000 samples')

    lead_data['III'] = (np.array(lead_data["II"]) - np.array(lead_data["I"]))
    lead_data['aVR'] = -(np.array(lead_data["I"]) + np.array(lead_data["II"]))/2
    lead_data['aVF'] = (np.array(lead_data["II"]) + np.array(lead_data["III"]))/2
    lead_data['aVL'] = (np.array(lead_data["I"]) - np.array(lead_data["III"]))/2

    lead_data = {k: lead_data[k] for k in lead_order}
    # drops V3R, V4R, and V7 if it was a 15-lead ECG

    temp = []
    for key,value in lead_data.items():
        temp.append(value)

    #transpose to be [time, leads, ]
    ekg_array = np.array(temp).T

    #expand dims to [time, leads, 1]
    ekg_array = np.expand_dims(ekg_array,  axis=-1)

    filename = '{}_{}.npy'.format(pt_id, AcquisitionDateTime)

    path_to_output += filename
    # print(path_to_output)
    with open(path_to_output, 'wb') as f:
        np.save(f, ekg_array)


def ekg_batch_run(ekg_list):
    i = 0
    x = 0
    for file in ekg_list:
        try:
            xml_to_np_array_file(file, output_dir)
            i+=1
        except Exception as e:
            # print("file failed: ", file)
            print(file, e)
            x+=1
        if i % 10000 == 0:
            print(f"Succesfully converted {i} EKGs, failed converting {x} EKGs")

output_dir = os.getcwd() + '/waveforms_output/'
print("args", sys.argv)
ekg_file_list = []
# file_path(sys.argv[1])  #if you want input to be a directory
file_path('data')  #if you want input to be a directory
print("Number of EKGs found: ", len(ekg_file_list))

ekg_batch_run(ekg_file_list)
