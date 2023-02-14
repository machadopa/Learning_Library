import pandas as pd
import sklearn.naive_bayes
import xgboost as xgb
import numpy as np
import sklearn.naive_bayes as nb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



def read_data(path:str)->pd.DataFrame:
    """

    :param path: pandas DataFrame
    :return: DataFrame with all the data
    """
    df = pd.read_csv(path)

    df = df.drop('date',axis=1)
    df = df.drop('month',axis=1)
    df = df.dropna(axis=1)
    return df

def split_data(df:pd.DataFrame):
    """
    Splits the data frame by features and labels
    :param df: pandas dataframe
    :return: features and labels train and test sets
    """
    X = df[df.columns[~df.columns.isin(['quarter'])]]  # Creating a data frame with only the features
    y = df[['quarter']]  # Creating a data frame with only the label(What I'm trying to predict)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5)
    return X_train,X_test,y_test,y_train


def trainer(X_train:pd.DataFrame,y_train:pd.DataFrame):
    """
    Practice for using set features to predict the data
    :param X_train: Features that will be used to train the model
    :param y_train: Labels that will be used to train the model
    :return: model
    """
    model = sklearn.naive_bayes.GaussianNB()
    model.fit(X_train,y_train.values.ravel(),sample_weight=None)
    return model
def possess(model,X_test:pd.DataFrame,Y_test:pd.DataFrame):
    '''

    :param model: Model
    :param X_test: features that will be tested
    :param Y_test: Labels that will be tested
    :return:
    '''
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    # Compare for accuracy
    accuracy = accuracy_score(y_test, predictions)
    return accuracy


def predict(features: pd.DataFrame) -> np.ndarray:

    features = features[features.columns]


if __name__ == "__main__":
    movies = read_data("C:\\Users\\FM Inventario\\OneDrive\Documents\\Movies Data Set.csv")

    X_train, X_test, y_train, y_test =split_data(movies)
    movie_model = trainer(X_train,y_train)
    accuracy_score = possess(movie_model,X_train,y_train)

    print(movies)
    print(accuracy_score)