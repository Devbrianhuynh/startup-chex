import pandas as pd
import csv


def load_dataset():
    startup_data = []

    with open('big_startup_secsees_dataset.csv', 'r') as file:
        reader = csv.DictReader(file)

        for row in reader:
            startup_data.append(row)

    startup_df = pd.DataFrame(data=startup_data).drop(columns=['permalink', 'homepage_url'])
    return startup_df


if __name__ == '__main__':
    load_dataset()