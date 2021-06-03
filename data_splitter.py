from sklearn.model_selection import train_test_split
import pandas as pd

if __name__ == "__main__":
    og_data = pd.read_csv('Dataset_crimes.csv')
    del og_data['Unnamed: 0']
    train_and_validation_data, test_data = train_test_split(og_data,
                                                            test_size=0.2,
                                                            train_size=0.8,
                                                            random_state=1925359273)
    train_data, validation_data = train_test_split(train_and_validation_data,
                                                   test_size=0.25,
                                                   train_size=0.75,
                                                   random_state=985484932)

    train_data.to_csv('train_dataset_crimes.csv')
    validation_data.to_csv('validation_dataset_crimes.csv')
    test_data.to_csv('test_dataset_crimes.csv')


