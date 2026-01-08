import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def stochastic_gradient_descent(data_path, num_epochs, batch_size, learning_rate):
    df = pd.read_csv(data_path)
    df = df.sample(frac=1).reset_index(drop=True)
    df['z_score_X'] = stats.zscore(df['X'])
    df['z_score_y'] = stats.zscore(df['y'])

    # Filter out outliers

    df= df[(df['z_score_X'].abs() < 3) & (df['z_score_y'].abs() < 3)]
    df = df.drop(columns=['z_score_X', 'z_score_y'])
    max_epochs=num_epochs

    n=df.shape[0]
    def compute_sums(df):
        sumx = df['X'].sum()
        sumy = df['y'].sum()
        sumx2 = (df['X'] * df['X']).sum()
        sumxy = (df['X'] * df['y']).sum()
        sumy2 = (df['y'] * df['y']).sum()
        return sumx, sumy, sumx2, sumxy, sumy2

#increasing batch size with epochs and decaying learning rate

    m = 1
    c = 1
    k = min(24,int(6000/batch_size))
    tol = 0.00001

    arr = []

    for i in range(0,k*batch_size,batch_size):
        df1=df.iloc[i:i+batch_size]
        n1 = df1.shape[0]
        sumx,sumy,sumx2,sumxy,sumy2=compute_sums(df1)
        J_last = (m*m*sumx2 + c*c*n1 + 2*c*m*sumx + sumy2 - 2*m*sumxy - 2*c*sumy) / (2*n1)
        temp_m = m - learning_rate * (m * sumx2 + c * sumx - sumxy) / n1
        c -= learning_rate * (m * sumx + n1 * c - sumy) / n1
        m = temp_m
        J1 = (m*m*sumx2 + c*c*n1 + 2*c*m*sumx + sumy2 - 2*m*sumxy - 2*c*sumy) / (2*n1)
        arr.append(abs(J1 - J_last))
        
    num = sum(arr)
    flag = 0
    for _ in range(int(max_epochs*batch_size/n)):
        #learning_rate/=(1+decay_rate*_)
        df2=df.sample(frac=1).reset_index(drop=True)
        for i in range(0,n,batch_size):
            df1=df2.iloc[i:i+batch_size]
            if num/k < tol:
               print("Epochs:",k+int(_*n/batch_size))
               flag=1
               break
            n1 = df1.shape[0]
            sumx,sumy,sumx2,sumxy,sumy2=compute_sums(df1)
            J_last = (m*m*sumx2 + c*c*n1 + 2*c*m*sumx + sumy2 - 2*m*sumxy - 2*c*sumy) / (2*n1)
            temp_m = m - learning_rate * (m * sumx2 + c * sumx - sumxy) / n1
            c -= learning_rate * (m * sumx + n1 * c - sumy) / n1
            m = temp_m
            J1 = (m*m*sumx2 + c*c*n1 + 2*c*m*sumx + sumy2 - 2*m*sumxy - 2*c*sumy) / (2*n1)
            arr.append(abs(J1 - J_last))
            num+=arr[-1]-arr[0]
            arr.pop(0)
            
        if flag==1:
            break


    return m, c


    
