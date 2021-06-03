import numpy as np
import math

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
    catches = 0
    for crime in crimes:
        if is_location_catch_crime(possible_location, crime):
            catches += 1

    return catches


def pick_best_location(possible_locations, crimes):
    max_catch_crimes = 0
    best_location = possible_locations[0]

    for cur_possible_location in possible_locations:
        cur_catch_crimes = get_catches_of_location(cur_possible_location, crimes)

        if cur_catch_crimes >= max_catch_crimes:
            max_catch_crimes = cur_catch_crimes
            best_location = cur_possible_location

    return best_location


def remove_close_crimes(crimes, cur_pick, num_of_days):
    close_indices = []

    for crime_index in range(len(crimes)):
        if is_location_catch_crime(cur_pick, crimes[crime_index]):
            close_indices += crime_index

    return np.delete(crimes, close_indices)


def fix_time(cur_pick):
    time = cur_pick[2]
    if time < BEFORE_AND_AFTER:
        time = BEFORE_AND_AFTER
    if time > MINUTES_PER_DAY - BEFORE_AND_AFTER:
        time = MINUTES_PER_DAY - BEFORE_AND_AFTER

    cur_pick[2] = time
    return cur_pick

def get_cars_locations(crimes, num_of_days = 0):
    possible_locations = get_possible_locations(crimes)
    cars_places_and_times = []

    for police_car_index in range(POLICE_CARS):
        cur_pick = pick_best_location(possible_locations, crimes)
        cur_pick = fix_time(cur_pick)
        cars_places_and_times += cur_pick
        crimes = remove_close_crimes(crimes, cur_pick, num_of_days)

    return cars_places_and_times


def verify_legal_time_intervals():
    assert 30 % TIME_INTERVALS == 0 and MINUTES_PER_DAY % TIME_INTERVALS == 0


if __name__ == '__main__':
    verify_legal_time_intervals()


