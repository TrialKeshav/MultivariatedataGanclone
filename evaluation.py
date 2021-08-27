import argparse
import numpy as np
import pandas as pd
from evalution_modules import Eval,Evalcombined
from sklearn.ensemble import AdaBoostRegressor
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--gendata_dir', type=str, default='./pretrained_models/finalSyntheticData.txt', help='Generated synthetic data directory',required=True)
    parser.add_argument('--sample_data', type=str, default='./example_data/sample_data.txt', help='Original data to be mimced',required=True)
    parser.add_argument('--n_estimators', type=int, default=125, help='Number of estimators of adaboostmodel',required=True)
    parser.add_argument('--random_state', type=int, default=42, help='Random state to considered by adaboost model',required=True)
    parser.add_argument('--bins', type=int, default=50, help='Number of bins for KDE plot', required=True)
    parser.add_argument('--pathsave', type=str, default='./pretrained_models/residuals_kdeplot',
                        help='Path and name to save residuals KDE plot', required=True)
    args = parser.parse_args()


    # Orginal data
    df0 = pd.read_csv(args.sample_data)
    X_train = df0.iloc[:, :-1].values
    y_train = df0.iloc[:, -1].values
    adbosstregressor0 = Eval(X_train, y_train, AdaBoostRegressor(random_state=args.random_state, n_estimators=args.n_estimators))
    adbosstregressor0.Train()
    _ = adbosstregressor0.predict()
    print('Mean Absolute Error of the original data modelled with Adaboost Regressor = {0}'.format(adbosstregressor0.MAE()))
    print('Mean Squared Error of the original data modelled with Adaboost Regressor = {0}'.format(adbosstregressor0.MSE()))
    print('Root Mean Squared Error of the original data modelled with Adaboost Regressor = {0}'.format(np.sqrt(adbosstregressor0.MSE())))
    print('Explained Variance Score of the original data modelled with Adaboost Regressor = {0}'.format(adbosstregressor0.Explained_Variance_Score()))
    print('\n')

    # Synthetic data
    df1 = pd.read_csv(args.gendata_dir)
    Xmm = df1.iloc[:, 1:-1].values
    ymm = df1.iloc[:, -1].values
    adbosstregressor = Eval(Xmm, ymm, AdaBoostRegressor(random_state=args.random_state, n_estimators=args.n_estimators))
    adbosstregressor.Train()
    _ = adbosstregressor.predict()
    print('Mean Absolute Error of the generated synthetic data modelled with Adaboost Regressor = {0}'.format(adbosstregressor.MAE()))
    print('Mean Squared Error of the generated synthetic data modelled with Adaboost Regressor = {0}'.format(adbosstregressor.MSE()))
    print('Root Mean Squared Error of the generated synthetic data modelled with Adaboost Regressor = {0}'.format(np.sqrt(adbosstregressor.MSE())))
    print('Explained Variance Score of the generated synthetic data modelled with Adaboost Regressor = {0}'.format(adbosstregressor.Explained_Variance_Score()))
    print('\n')

    # Combined Data
    ARcom = Evalcombined(indipendentvariable=X_train, dependentvariable=y_train,
                         model=AdaBoostRegressor(random_state=args.random_state, n_estimators=args.n_estimators))
    ARcom.Train(Xmm=Xmm, ymm=ymm)
    _ = ARcom.predict()
    print('Mean Absolute Error of the combined original and generated synthetic data modelled with Adaboost Regressor = {0}'.format(ARcom.MAE()))
    print('Mean Squared Error of the combined original and generated synthetic data modelled with Adaboost Regressor = {0}'.format(ARcom.MSE()))
    print('Root Mean Squared Error of combined original and generated synthetic data modelled with Adaboost Regressor = {0}'.format(np.sqrt(ARcom.MSE())))
    print('Explained Variance Score of combined original and generated synthetic data modelled with Adaboost Regressor = {0}'.format(ARcom.Explained_Variance_Score()))

    #Plotting Residuals distribution
    ARcom.Errorkdeplot(args.pathsave,args.bins)

















