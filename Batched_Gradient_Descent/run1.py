import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def gradient_descent(data_path, num_epochs, batch_size, learning_rate):
    df = pd.read_csv(data_path)
    df = df.sample(frac=1).reset_index(drop=True)
    df['z_score_X'] = stats.zscore(df['X'])
    df['z_score_y'] = stats.zscore(df['y'])

    # Filter out outliers

    df= df[(df['z_score_X'].abs() < 3) & (df['z_score_y'].abs() < 3)]
    df = df.drop(columns=['z_score_X', 'z_score_y'])
    max_epochs=num_epochs

    n=df.shape[0]
    sumx= df['X'].sum()
    sumy= df['y'].sum()
    sumx2=(df['X'] * df['X']).sum()
    sumxy=(df['X'] * df['y']).sum()
    sumy2=(df['y'] * df['y']).sum()


    def J(m,c,n):
        return (m*m*sumx2+c*c*n+2*c*m*sumx+sumy2-2*m*sumxy-2*c*sumy)/(2*n)

    def main(learn_rate, tol, max_epochs):
        m = 1
        c = 1
        J_prev = 0
        J_curr = J(m, c, n)
        for i in range(max_epochs):
            temp_m = m - learn_rate * (m * sumx2 + c * sumx - sumxy) / n
            c -= learn_rate * (m * sumx + n * c - sumy) / n
            m = temp_m
            J_prev = J_curr
            J_curr = J(m, c, n)
            if abs(J_curr - J_prev) < tol:
                print("Epochs:", i)
                break
        return m, c

    m, c = main(learning_rate, 0.00001, max_epochs)

    return m,c
    
    