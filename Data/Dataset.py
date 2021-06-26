import torch
import torch.utils.data
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle

def load_dataloader(tabular_data,batch_size,args):
    obtain_shape = tabular_data.shape
    samples = int(obtain_shape[0] / batch_size)
    train_labels = torch.zeros((samples * batch_size, 1),device=torch.device(args.device))  # No of zeros
    train_set = [(tabular_data[i, :], train_labels[i]) for i in range(samples * batch_size)]
    return torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)


def make_train_data(args,encodedDataframe):
    #dataset = pd.read_csv(args.sample_data)
    Tabular_data = torch.tensor(encodedDataframe.iloc[:, :].values)
    #Scaling the data for future predictions
    sc = StandardScaler()
    Tabular_data = sc.fit_transform(Tabular_data)
    #Diemensionality reduction to avoid overfitting
    pca = PCA()
    Tabular_data = pca.fit_transform(Tabular_data)

    standardscaler_file = "standard_scaler.pickle"
    pickle.dump(sc, open(args.output_dir + standardscaler_file, 'wb'))
    pca_file = "pca.pickle"
    pickle.dump(pca, open(args.output_dir + pca_file, 'wb'))
    return load_dataloader(tabular_data=Tabular_data, batch_size=args.batch_size,args = args)