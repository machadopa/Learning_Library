import pandas as pd
import xgboost as xgb
import numpy as np



def read_data(path:str)->pd.DataFrame:
    df = pd.read_csv(path)
    return df

def classifier(features: pd.DataFrame,labels: pd.Series) -> None:

    np.random.seed(3)
    columns = np.random.choice(features.columns, 10,replace=False) # Is there where we create a new DF with only classifiers
    print(columns)
    features = features[columns] #What should this represent?

    model = xgb.XGBRegressor()
    model.fit(features,labels) #I'm not sure what this line of code is for. Should I put it in a train funcion?

def predict(features: pd.DataFrame) -> np.ndarray:

    features = features[features.columns]


if __name__ == "__main__":
    movies = read_data("C:\\Users\\FM Inventario\\OneDrive\Documents\\Movies Data Set.csv")

    print(classifier(movies,movies.columns)) #I'm getting an error here




