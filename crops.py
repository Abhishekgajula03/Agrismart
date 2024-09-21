import numpy as np 
import pandas as pd

crop=pd.read_csv("Crop_recommendation.csv")
print(crop.head())
print("------------------------------------------------")
print(crop.shape)
print("------------------------------------------------")
print(crop.info())

print(crop.isnull().sum())
print("------------------------------------------------")
print(crop.duplicated().sum())
print("------------------------------------------------")
print(crop.describe())

corr=crop.corr() 
print(corr)
