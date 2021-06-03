import numpy as np
import math
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import datetime

# The maximum distance from police to crime
MAX_DISTANCE = 500

# NUMBER OF MINUTES BETWEEN EACH TRY
# (60 * 24) % TIME_INTERVALS should be 0
# 30 % TIME_INTERVALS should be 0
TIME_INTERVALS = 10

# NUMBER OF POLICE CARS
POLICE_CARS = 30
MINUTES_PER_DAY = 60 * 24

# Time interval that counts as a catch
BEFORE_AND_AFTER = 30


def get_possible_locations(crimes):
    # kmeans = KMeans(n_clusters=3000, random_state=0).fit(crimes)
    # centers = kmeans.cluster_centers_
    # print(centers.shape)
    return crimes


def get_distance(X1, Y1, X2, Y2):
    x_dif = X1 - X2
    y_dif = Y1 - Y2

    return math.sqrt(x_dif ** 2 + y_dif ** 2)


def is_location_catch_crime(possible_location, crime):
    location_time = possible_location[2]
    crime_time = crime[2]

    is_in_time = (crime_time - BEFORE_AND_AFTER <= location_time <= crime_time + BEFORE_AND_AFTER)
    is_in_distance = (get_distance(possible_location[0], possible_location[1], crime[0], crime[1]) <= MAX_DISTANCE)

    return is_in_time and is_in_distance


def get_catches_of_location(possible_location, crimes):
    # catches = 0
    # for crime in crimes:
    #     if is_location_catch_crime(possible_location, crime):
    #         catches += 1
    #
    # if (len(get_indices_of_catches(possible_location, crimes)) != catches):
    #     print("SHITTT")
    # else:
    #     print("GOOD")


    return len(get_indices_of_catches(possible_location, crimes))


def get_indices_of_catches(location, crimes):
    Xs = crimes[:,0]
    Ys = crimes[:,1]
    times = crimes[:,2]

    distances = np.sqrt(((Xs - location[0]) ** 2) + ((Ys - location[1]) ** 2))
    close_distances_indexes = np.where(distances <= 500)
    close_times_indexes = np.where((times >= location[2] - BEFORE_AND_AFTER) & (times <= location[2] + BEFORE_AND_AFTER))

    return np.intersect1d(close_distances_indexes, close_times_indexes)
    # return close_distances_indexes


def pick_best_location(possible_locations, crimes):
    max_catch_crimes = 0
    best_location = possible_locations[0]

    for cur_possible_location in possible_locations:
        cur_catch_crimes = get_catches_of_location(cur_possible_location, crimes)

        if cur_catch_crimes >= max_catch_crimes:
            max_catch_crimes = cur_catch_crimes
            best_location = cur_possible_location

    print(f"best location time: {best_location[2] / 60} : {best_location[2] % 60}")

    print("MATCH LOCATIONS")
    print(max_catch_crimes)
    return best_location


def remove_close_crimes(crimes, cur_pick, num_of_days):
    close_indices = get_indices_of_catches(cur_pick, crimes)

    # for crime_index in range(len(crimes)):
    #     if is_location_catch_crime(cur_pick, crimes[crime_index]):
    #         close_indices.append(crime_index)

    after_del = np.delete(crimes, close_indices, axis=0)
    return after_del


def fix_time(cur_pick):
    time = cur_pick[2]
    if time < BEFORE_AND_AFTER:
        time = BEFORE_AND_AFTER
    if time > MINUTES_PER_DAY - BEFORE_AND_AFTER:
        time = MINUTES_PER_DAY - BEFORE_AND_AFTER

    cur_pick[2] = time
    return cur_pick


def get_cars_locations(crimes, num_of_days=0):
    possible_locations = get_possible_locations(crimes)
    cars_places_and_times = np.empty([0, 3])

    for police_car_index in range(POLICE_CARS):
        print("CUR POLICE")
        print(crimes.shape)
        cur_pick = pick_best_location(possible_locations, crimes)
        cur_pick = fix_time(cur_pick)
        cars_places_and_times = np.vstack([cars_places_and_times, cur_pick])
        crimes = remove_close_crimes(crimes, cur_pick, num_of_days)
        print(f"FOUND LOCATION {police_car_index}")

    return cars_places_and_times


def verify_legal_time_intervals():
    assert 30 % TIME_INTERVALS == 0 and MINUTES_PER_DAY % TIME_INTERVALS == 0


def get_all_train_data():
    data = pd.read_csv('../train_dataset_crimes.csv')
    data['Datetime'] = pd.to_datetime(data['Date'], format='%m/%d/%Y %I:%M:%S %p')
    data['Datetime'] = data['Datetime'].dt.hour * 60 + data['Datetime'].dt.minute
    return data[['X Coordinate', 'Y Coordinate', 'Datetime']].dropna().to_numpy()

def get_all_train_data_of_weekday(date):
    data = pd.read_csv('../train_dataset_crimes.csv')
    data['Datetime'] = pd.to_datetime(data['Date'], format='%m/%d/%Y %I:%M:%S %p')
    data = data[data["Datetime"].apply(lambda x: x.weekday() == date.weekday())]

    data['Datetime'] = data['Datetime'].dt.hour * 60 + data['Datetime'].dt.minute
    return data[['X Coordinate', 'Y Coordinate', 'Datetime']].dropna().to_numpy()


def get_data_of_one_date(random_date):
    data = pd.read_csv('../validation_dataset_crimes.csv')
    data['Datetime'] = pd.to_datetime(data['Date'], format='%m/%d/%Y %I:%M:%S %p')

    data = data[data["Datetime"].apply(lambda x: x.date() == random_date)]

    data['Datetime'] = pd.to_datetime(data['Date'], format='%m/%d/%Y %I:%M:%S %p')
    data['Datetime'] = data['Datetime'].dt.hour * 60 + data['Datetime'].dt.minute
    return data[['X Coordinate', 'Y Coordinate', 'Datetime']].dropna().to_numpy()

def get_random_date():
    data = pd.read_csv('../validation_dataset_crimes.csv')
    data['Datetime'] = pd.to_datetime(data['Date'], format='%m/%d/%Y %I:%M:%S %p')
    random_date = data.sample().iloc[0]['Datetime'].date()

    return random_date


def plot_crimes_and_locations(crimes, locations):
    crimes = crimes[:,:2]
    locations = locations[:,:2]
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.scatter(crimes[:,0], crimes[:,1])
    plt.scatter(locations[:,0], locations[:,1])
    plt.show()


def print_crimes_times(param):
    plt.hist(param/60, 24)
    plt.show()


def get_test_data(size):
    data = pd.read_csv('../test_dataset_crimes.csv')
    data['Datetime'] = pd.to_datetime(data['Date'], format='%m/%d/%Y %I:%M:%S %p')
    data['Datetime'] = pd.to_datetime(data['Date'], format='%m/%d/%Y %I:%M:%S %p')
    data['Datetime'] = data['Datetime'].dt.hour * 60 + data['Datetime'].dt.minute
    data = data[['X Coordinate', 'Y Coordinate', 'Datetime']].dropna().to_numpy()
    np.random.shuffle(data)
    return data[:size,:]


if __name__ == '__main__':
    path = "locations.npy"
    date = get_random_date()
    # date = datetime.datetime.strptime('06052021', '%d%m%Y').date()

    # crimes = get_all_train_data_of_weekday(date)
    crimes = get_all_train_data()

    verify_legal_time_intervals()
    locations = get_cars_locations(crimes)
    np.save(path, locations)

    locations = np.load(path)

    # plot_crimes_and_locations(crimes, locations)


    # validation_crimes = get_data_of_one_date(date)
    validation_crimes = get_test_data(400)
    print(validation_crimes)
    print(len(validation_crimes))

    catches = np.array([])
    for location in locations:
        cur_catches = get_indices_of_catches(location, validation_crimes)
        catches = np.union1d(catches, cur_catches)

    catches = np.unique(catches)
    print(catches)
