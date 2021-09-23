# This file reads and cleans the data of the Human Development Index


import json
import csv
import pandas
import pycountry


def clean_csv(file):
    '''
    Matches an Alpha-3 code to a country
    '''
    # read csv file with pandas
    reader = pandas.read_csv(file, delimiter=';')
    df = pandas.DataFrame(data=reader)
    # loop through csv
    for index, row in df.iterrows():
        # for every row, find alpha-3 code that matches country's name
        code = pycountry.countries.get(name=row["country"]).alpha_3
        # add alpha-3 code to that specific row
        df.iloc[index,0] = code
    # make new csv file which is used during the project
    df.to_csv("output.csv")


if __name__ == '__main__':
    clean_csv('Human Development Index (HDI).csv')
