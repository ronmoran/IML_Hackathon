import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from datetime import datetime
import numpy as np

crimes_dict = {0: 'BATTERY', 1: 'THEFT', 2: 'CRIMINAL DAMAGE', 3: 'DECEPTIVE PRACTICE', 4: 'ASSAULT'}
crimes_inv_dict = {v: k for k, v in crimes_dict.items()}
LOCATION_DESCRIPTION_COL = "Location Description"
labels_col = "Primary Type"
COLS_FOR_FREQ = ["Ward", "Beat", "District", "Community Area"]


def pre_processing(X: pd.DataFrame):

    processed_data = X[['Date', 'Arrest', 'Domestic', 'X Coordinate',
                        'Y Coordinate', 'Updated On', "Beat", "District",
                        "Ward", "Community Area",
                        LOCATION_DESCRIPTION_COL]].copy()
    processed_data['Date'] = pd.to_datetime(processed_data['Date'], format='%m/%d/%Y %I:%M:%S %p')
    processed_data['Updated On'] = pd.to_datetime(processed_data['Updated On'], format='%m/%d/%Y %I:%M:%S %p')
    processed_data['Time diff'] = processed_data['Updated On'] - processed_data['Date']
    # processed_data[LOCATION_DESCRIPTION_COL] = processed_data[
    #     LOCATION_DESCRIPTION_COL].apply(lambda x: LOCATION_DESCRIPTION.get(
    #     x, -1))
    processed_data['Time diff'] = processed_data['Time diff'].apply(lambda x: x.total_seconds())
    processed_data['Hour'] = processed_data['Date'].apply(lambda x: x.time().hour)
    processed_data['Weekday'] = processed_data['Date'].apply(lambda x: x.date().weekday())
    for col in COLS_FOR_FREQ:
        tab = pd.read_pickle(f"{col}.pkl")
        indices = processed_data[col].unique()
        to_append = tab.reindex(indices)
        to_append = to_append.loc[processed_data[col].values, :]
        fill_vals = to_append.dropna(axis=0).mean(axis=0)
        to_append.fillna(fill_vals, axis=0, inplace=True)
        processed_data.loc[:, to_append.columns.values] = to_append.values
    mean_coor = processed_data[['X Coordinate', 'Y Coordinate']].dropna(
        axis=0).mean(axis=0)
    processed_data.fillna(mean_coor, axis=0, inplace=True)
    processed_data.fillna({LOCATION_DESCRIPTION_COL: " "}, axis=0, inplace=True)
    processed_data.fillna({"Arrest": False, "Domestic": False}, axis=0,
                          inplace=True)
    # tab = pd.read_pickle(f"{LOCATION_DESCRIPTION_COL}.pkl")
    # words = processed_data[LOCATION_DESCRIPTION_COL].str.findall("["
    #                                                              "A-Za-z0-9]+")
    processed_data.drop(['Date', 'Updated On', *COLS_FOR_FREQ],
                        axis=1, inplace=True)
    if labels_col in processed_data.columns:
        processed_data.drop([labels_col], inplace=True, axis=1)
    processed_data.dropna(axis=0, inplace=True)
    return processed_data


def get_col_freq(col: str, labels_col:str, data: pd.DataFrame) -> pd.DataFrame:
    freq = pd.crosstab(data[col], data[labels_col])
    freq = freq.div(freq.sum(axis=1), axis=0)
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


def train(x: pd.DataFrame, y: pd.DataFrame):
    clf = CatBoostClassifier(iterations=400, depth=4,
                             cat_features=['Weekday', 'Hour'],
                             text_features=[LOCATION_DESCRIPTION_COL])
    # clf = RandomForestClassifier(n_estimators=1500, max_depth=11,
    #                              max_features='auto')
    clf.fit(x, y)
    dump_model(clf, "Q1_model.pkl")
    return clf


def dump_model(model: CatBoostClassifier, path: str):
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load_model(path) -> CatBoostClassifier:
    with open(path, 'rb') as f:
        return pickle.load(f)


def predict(X):
    cb = load_model("Q1_model.pkl")
    X_df = pd.read_csv(X)
    prediction = cb.predict(pre_processing(X_df)).flatten()
    return [crimes_dict[ind] for ind in prediction]
def send_police_cars(X):
    locations = np.load('Q2_model.npy')
    new_date = datetime.strptime(X, '%m/%d/%Y %I:%M:%S %p')
    all_dates = []
    for i in range(locations.shape[0]):
        i_hours = np.int64(locations[i, 2] // 60)
        i_minutes = np.int64(locations[i, 2] % 60)
        i_date = new_date.replace(hour=i_hours, minute=i_minutes)
        all_dates.append(i_date.strftime('%m/%d/%Y %I:%M:%S %p'))
    return [(locations[i, 0], locations[i, 1], all_dates[i]) for i in range(locations.shape[0])]

