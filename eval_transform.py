import pandas as pd
import pickle


def preprocessingFirstColumnSimilar_train(label_encoderFile,onehot_encoderFile,csv_path):

    #Orginal data
    local_labelencoded = pickle.load(open(label_encoderFile, 'rb'))
    local_onehot_encoder = pickle.load(open(onehot_encoderFile, 'rb'))
    dataset = pd.read_csv(csv_path)
    integer_encoded = local_labelencoded.transform(dataset['Cell_ID'].values)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = local_onehot_encoder.transform(integer_encoded)

    #Make a new dataframe with corresponding column names equivaluent to cell_id
    #Here we have Cell ID name seuqntial from 3 to 23
    onehot_cellidDataframe = pd.DataFrame(onehot_encoded,columns=["3", '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                                '19', '20', '21', '22', '23'])

    #Adding back the remaining columns
    onehot_cellidDataframe["Velocity"] = dataset["Velocity"]
    onehot_cellidDataframe['Load'] = dataset["Load"]
    onehot_cellidDataframe['EdgeUsersDist'] = dataset["EdgeUsersDist"]
    onehot_cellidDataframe['Hysteresis'] = dataset["Hysteresis"]
    onehot_cellidDataframe['TTT'] = dataset["TTT"]
    onehot_cellidDataframe['CIO(3)'] = dataset["CIO(3)"]
    onehot_cellidDataframe['CIO(4)'] = dataset["CIO(4)"]
    onehot_cellidDataframe['CIO(5)'] = dataset["CIO(5)"]
    onehot_cellidDataframe['CIO(6)'] = dataset["CIO(6)"]
    onehot_cellidDataframe['CIO(7)'] = dataset["CIO(7)"]
    onehot_cellidDataframe['CIO(8)'] = dataset["CIO(8)"]
    onehot_cellidDataframe['CIO(9)'] = dataset["CIO(9)"]
    onehot_cellidDataframe['CIO(10)'] = dataset["CIO(10)"]
    onehot_cellidDataframe['CIO(11)'] = dataset["CIO(11)"]
    onehot_cellidDataframe['CIO(12)'] = dataset["CIO(12)"]
    onehot_cellidDataframe['CIO(13)'] = dataset["CIO(13)"]
    onehot_cellidDataframe['CIO(14)'] = dataset["CIO(14)"]
    onehot_cellidDataframe['CIO(15)'] = dataset["CIO(15)"]
    onehot_cellidDataframe['CIO(16)'] = dataset["CIO(16)"]
    onehot_cellidDataframe['CIO(17)'] = dataset["CIO(17)"]
    onehot_cellidDataframe['CIO(18)'] = dataset["CIO(18)"]
    onehot_cellidDataframe['CIO(19)'] = dataset["CIO(19)"]
    onehot_cellidDataframe['CIO(20)'] = dataset["CIO(20)"]
    onehot_cellidDataframe['CIO(21)'] = dataset["CIO(21)"]
    onehot_cellidDataframe['CIO(22)'] = dataset["CIO(22)"]
    onehot_cellidDataframe['CIO(23)'] = dataset["CIO(23)"]
    onehot_cellidDataframe['GKPI'] = dataset["GKPI"]

    return onehot_cellidDataframe