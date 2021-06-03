import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

crimes_dict = {0: 'BATTERY', 1: 'THEFT', 2: 'CRIMINAL DAMAGE', 3: 'DECEPTIVE PRACTICE', 4: 'ASSAULT'}
crimes_inv_dict = {v: k for k, v in crimes_dict.items()}  # todo return crimes and not label
LOCATION_DESCRIPTION = {}
LOCATION_DESCRIPTION_COL = "Location Description"
labels_col = "Primary Type"
COLS_FOR_FREQ = [
                 LOCATION_DESCRIPTION_COL]


def pre_processing(X: pd.DataFrame):

    processed_data = X[['Date', 'Arrest', 'Domestic', 'X Coordinate',
                        'Y Coordinate', 'Updated On', "Beat", "District",
                        "Ward", "Community Area",
                        LOCATION_DESCRIPTION_COL]].copy()
    processed_data['Date'] = pd.to_datetime(processed_data['Date'], format='%m/%d/%Y %I:%M:%S %p')
    processed_data['Updated On'] = pd.to_datetime(processed_data['Updated On'], format='%m/%d/%Y %I:%M:%S %p')
    processed_data['Time diff'] = processed_data['Updated On'] - processed_data['Date']
    processed_data['Time diff'] = processed_data['Time diff'].apply(lambda x: x.total_seconds())
    processed_data['Hour'] = processed_data['Date'].apply(lambda x: x.time().hour)
    processed_data['Weekday'] = processed_data['Date'].apply(lambda x: x.date().weekday())
    for col in COLS_FOR_FREQ:
        tab = pd.read_pickle(f"{col}.pkl")
        to_append: pd.DataFrame = tab.loc[processed_data[col].values, :]
        processed_data.loc[:, to_append.columns.values] = to_append.values
    # tab = pd.read_pickle(f"{LOCATION_DESCRIPTION_COL}.pkl")
    # words = processed_data[LOCATION_DESCRIPTION_COL].str.findall("["
    #                                                              "A-Za-z0-9]+")
    processed_data.drop(['Date', 'Updated On', *COLS_FOR_FREQ,
                         LOCATION_DESCRIPTION_COL],
                        axis=1, inplace=True)
    return processed_data


def get_col_freq(col: str, labels_col:str, data: pd.DataFrame) -> pd.DataFrame:
    freq = pd.crosstab(data[col], data[labels_col])
    freq:pd.DataFrame = freq.div(freq.sum(axis=1), axis=0)
    freq.rename(columns={label_name: f"{col}_{label_name}" for label_name in
                         freq.columns}, inplace=True)
    return freq


def get_word_freq(col: pd.Series, labels:pd.Series) -> pd.DataFrame:
    words = col.str.findall("[A-Za-z0-9]+")
    cols = [f"{LOCATION_DESCRIPTION_COL}_{label}" for label
            in labels.unique()]
    freq_tab = pd.DataFrame(columns=cols)
    for row, label in zip(words, labels):
        for word in row:
            if word not in freq_tab.index:
                freq_tab.loc[word] = pd.Series(index=cols,
                                               data=[0 for _ in cols])
            freq_tab.loc[word, f"{LOCATION_DESCRIPTION_COL}_{label}"] += 1
    return freq_tab.div(freq_tab.sum(axis=1), axis=0)


def train_pre_process(X: pd.DataFrame):
    processed_data = X.dropna()
    y_train = processed_data['Primary Type'].apply(lambda x: crimes_inv_dict[x])
    for col in COLS_FOR_FREQ:
        tab = get_col_freq(col, labels_col, processed_data)
        tab.to_pickle(f"{col}.pkl")
    # word_freq = get_word_freq(processed_data[LOCATION_DESCRIPTION_COL],
    #                           y_train)
    # word_freq.to_pickle(f"{LOCATION_DESCRIPTION_COL}.pkl")
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
