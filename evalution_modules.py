import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score


class Eval():

    def __init__(self,indipendentvariable,dependentvariable,model):

        self.indipendentvariable = indipendentvariable
        self.dependentvariable = dependentvariable
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(indipendentvariable, dependentvariable, test_size=0.33, random_state=42)

        self.pipe = make_pipeline(MinMaxScaler(),model)


    def Train(self):

        self.pipe.fit(self.X_train,self.y_train)


    def predict(self):

        self.ypred = self.pipe.predict(self.X_test)
        return  self.ypred


    def MAE(self):
        self.mae = mean_absolute_error(self.y_test,self.ypred)
        return self.mae

    def MSE(self):
        self.mse = mean_squared_error(self.y_test,self.ypred)
        return self.mse

    def Explained_Variance_Score(self):
        self.evs = explained_variance_score(self.y_test,self.ypred)
        return self.evs

    def Errorkdeplot(self,path,bins):
        sns.distplot((self.y_test - self.ypred), bins=bins)
        plt.xlabel('Residuals')
        plt.savefig(path+'.png')



class Evalcombined(Eval):

    def Train(self,Xmm,ymm):

        self.Xcom = np.concatenate((self.X_train,Xmm))
        self.ycom = np.concatenate((self.y_train,ymm))
        self.pipe.fit(self.Xcom,self.ycom)
