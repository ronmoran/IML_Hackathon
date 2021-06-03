import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

crimes_dict = {0: 'BATTERY', 1: 'THEFT', 2: 'CRIMINAL DAMAGE', 3: 'DECEPTIVE PRACTICE', 4: 'ASSAULT'}
crimes_inv_dict = {v: k for k, v in crimes_dict.items()}  # todo return crimes and not label
LOCATION_DESCRIPTION = {}
LOCATION_DESCRIPTION_COL = "Location Description"


def pre_processing(X: pd.DataFrame):
    processed_data = X[['Date', 'Arrest', 'Domestic', 'X Coordinate',
                        'Y Coordinate', 'Updated On', "Beat",
                        LOCATION_DESCRIPTION_COL]].copy()

    processed_data['Date'] = pd.to_datetime(processed_data['Date'], format='%m/%d/%Y %I:%M:%S %p')
    processed_data['Updated On'] = pd.to_datetime(processed_data['Updated On'], format='%m/%d/%Y %I:%M:%S %p')
    processed_data['Time diff'] = processed_data['Updated On'] - processed_data['Date']
    processed_data[LOCATION_DESCRIPTION_COL] = processed_data[
        LOCATION_DESCRIPTION_COL].apply(lambda x: LOCATION_DESCRIPTION.get(
        x, -1))  # todo why -1? what to do with newly found data
    processed_data['Time diff'] = processed_data['Time diff'].apply(lambda x: x.total_seconds())
    processed_data['Hour'] = processed_data['Date'].apply(lambda x: x.time().hour)
    processed_data['Weekday'] = processed_data['Date'].apply(lambda x: x.date().weekday())
    processed_data = pd.concat((processed_data, pd.get_dummies(processed_data['Weekday'],
                                               prefix="weekday_")), 1)

    processed_data.drop(['Date', 'Updated On', "Weekday"], axis=1,
                        inplace=True)
    return processed_data


def train_pre_process(X: pd.DataFrame):
    processed_data = X.dropna()
    y_train = processed_data['Primary Type'].apply(lambda x: crimes_inv_dict[x])
    global LOCATION_DESCRIPTION
    if LOCATION_DESCRIPTION == {}:
        LOCATION_DESCRIPTION = {val: i for i, val in enumerate(set(
                processed_data[LOCATION_DESCRIPTION_COL]))}
    return pre_processing(processed_data), y_train


def train(x: pd.DataFrame, y:pd.DataFrame, depth=10) -> RandomForestClassifier:
    clf = RandomForestClassifier(n_estimators=200, max_depth=depth,
                                 max_features=0.75,
                                 random_state=0)
    clf.fit(x, y)
    return clf


def dump_model(model: RandomForestClassifier, path: str):
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load_model(path) -> RandomForestClassifier:
    with open(path, 'rb') as f:
        return pickle.load(f)


def predict(X, rf: RandomForestClassifier): #todo change back to X only
    return rf.predict(pre_processing(X))


def send_police_cars(X):
    pass
