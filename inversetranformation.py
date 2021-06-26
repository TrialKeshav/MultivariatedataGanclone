import torch
import numpy as np

def revert_encoding(arr,label_encoder):
    temp = list()
    for j in range(arr.shape[0]):
        temp.append(label_encoder.inverse_transform([np.argmax(arr[j, :])]))
    return np.array(temp)

