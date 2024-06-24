import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import pandas as pd
from extractdata import extractdata_faculty
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import csv


class FirstModel(nn.Module):

    def __init__(self, subjectcount, y):
        super().__init__()

        self.A = nn.Parameter(torch.randn(subjectcount) + y.mean(), requires_grad=True)
        self.U = nn.Parameter(torch.randn(subjectcount) + y.mean(), requires_grad=True)
        self.Lambda = nn.Parameter(torch.rand(subjectcount), requires_grad=True)
        self.Gamma1 = nn.Parameter(torch.rand(subjectcount), requires_grad=True)

    def modify_params(self):
        return F.relu(self.A), F.relu(self.U), F.sigmoid(self.Lambda)*.2, F.sigmoid(self.Gamma1)

    @staticmethod
    def compute_mu(A, U, Lambda, Gamma1, sub, j, k):
        mu = A[sub] - U[sub] * (torch.exp(-Lambda[sub] * (j + Gamma1[sub] * k)))
        return mu

    def forward(self, y, j, k, sub):
        length = y.shape[0]
        mu = torch.zeros(length)
        A, U, Lambda, Gamma1 = self.modify_params()
        mu = self.compute_mu(A, U, Lambda, Gamma1, sub, j, k)
        loss = torch.sqrt((y - mu).pow(2).mean())
        return loss


def constrain_A_U(model):
    model.A.data = torch.max(model.A.data, model.U.data)


def mle(subjectcount, y, lr):
    model = FirstModel(subjectcount, y)
    optimiser = optim.Adam(model.parameters(), lr=lr)
    return model, optimiser


def evaluate_predictions(mupredicted, y, k, n):
    mse = mean_squared_error(y, mupredicted)
    rmse = np.sqrt(mean_squared_error(y, mupredicted))
    mae = mean_absolute_error(y, mupredicted)
    r2 = r2_score(y, mupredicted)
    aic = 2*k+n*np.log(mse)
    bic = k*np.log(n)+n*np.log(mse)
    print("MSE", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2:", r2)
    print("AIC:", aic)
    print("BIC:", bic)
    return mse, rmse, mae, r2, aic, bic


if __name__ == '__main__':

    file_name = "Game1_qmatrix_alldata"
    df = pd.read_csv(file_name + ".csv")
    y, j, k1, k2, k3, k4, k5, k6, sub, subjectcount = extractdata_faculty(df)

    n = 30000
    lr = .01
    model, opt = mle(subjectcount, y, lr)

    for i in range(n):
        loss = model(y, j, k1, sub)
        opt.zero_grad() 
        loss.backward()
        if i % 5000 == 0:
            print('Loss: ', loss.item())
        opt.step()
        constrain_A_U(model)

    A_out, U_out, Lambda_out, Gamma1_out = model.modify_params()
    
    mupredicted = FirstModel.compute_mu(A_out, U_out, Lambda_out, Gamma1_out,  sub, j, k1).detach()

    k = 4  # k:number of parameters for Model2
    n_s = 22179  # number of user samples for Game1
    evaluate_predictions(mupredicted, y, k, n_s)

