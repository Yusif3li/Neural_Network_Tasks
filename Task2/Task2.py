import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


Features = ["CulmenLength", "CulmenDepth", "FlipperLength", "OriginLocation", "BodyMass"]
classes = ["Adelie", "Chinstrap", "Gentoo"]

def preprocessing_data():
    data = pd.read_csv("penguins.csv")
    target=data["Species"]
    data = data.drop("Species", axis=1)
    
    print(data.info)
    print(data.isnull().sum())
    print(data.dtypes)
    print(data.duplicated().sum())

    numeric_data = data.select_dtypes(include=["number"])
    categorical_data = data.drop(columns=numeric_data.columns)

    numeric_data=numeric_data.fillna(numeric_data.mean())
    categorical_data=categorical_data.fillna(categorical_data.mode().iloc[0])

    data=pd.concat([numeric_data,categorical_data], axis=1)

    encoder=LabelEncoder()
    data["OriginLocation"] = encoder.fit_transform(data["OriginLocation"])
    data["Species"] = target
    return data