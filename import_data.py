#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 16:51:08 2021

@author: jonathanharris
"""

import os
import numpy as np
import pandas as pd
import re


def hdi_classification(score):
    '''
    Convert hdi score to classification as defined by http://hdr.undp.org/

    Parameters
    ----------
    score : float

    Returns
    -------
    classification
    '''

    if score < 0.550:
        return 'low'
    elif score < 0.7:
        return 'medium'
    elif score < 0.8:
        return 'high'
    elif score >= 0.8:
        return 'greater'

    return np.nan


def import_merge():
    '''
    import relevant files, process primary keys, and merge datasets

    Parameters
    ----------
    rootpath : str

    Returns
    -------
    dataframe

    '''

    # === IMPORT SPEECHES ===
    sessions = np.arange(25, 76)
    data = []

    for session in sessions:
        directory = "data/speeches/Session "+str(session)+" - "+str(1945+session)

        for filename in os.listdir(directory):
            f = open(os.path.join(directory, filename))

            if filename[0] == ".":  # ignore hidden files
                continue
            splt = filename.split("_")
            data.append([session, 1945+session, splt[0], f.read()])

    df_speech = pd.DataFrame(data, columns=['Session', 'Year', 'ISO-alpha3 Code', 'Speech'])

    # === IMPORT ACCESSORY DATASETS ===
    df_code = pd.read_csv('data/country_codes.csv')
    df_happy = pd.read_excel('data/country_happiness_metrics.xls')
    df_happy.rename(columns={'year': 'Year'}, inplace=True)  # for merge later

    df_hdi = pd.read_csv('data/human_development_index.csv', sep=';')
    df_hdi = pd.melt(df_hdi, id_vars=['Country'], value_vars=df_hdi.columns[2:])
    df_hdi.columns = ['Country', 'Year', 'hdi_score']
    df_hdi['Year'] = df_hdi['Year'].apply(int)
    df_hdi['Country'] = df_hdi['Country'].str.strip()
    df_hdi['hdi_score'].replace('..', np.nan, inplace=True)
    df_hdi['hdi_score'] = df_hdi['hdi_score'].apply(float)

    # === CLEAN UP KEYS PRIOR TO MERGE ===
    # use df_code as primary key -- inspect for weird characters
    to_replace = [s for s in df_code['Country or Area'] if re.search(r'[À-ÿ]+', s)]
    new_countries = ['Reunion', "Cote d'Ivoire", 'Curacao', 'Saint Barthelemy', 'Aland Island']
    replace_dict = dict(zip(to_replace, new_countries))
    df_code['Country or Area'].replace(replace_dict, inplace=True)
    df_code['Country or Area'].replace({'Viet Nam': 'Vietnam'}, inplace=True)

    # prepare df_speed key to merge with df_code
    # EU -- unsure what to do
    # CSK -- Czechoslovakia divided into Czechia (CZE), and Slovakia (SVK)
    # YUG -- divided into multiple countries
    speech_replace = {'POR': 'PRT',
                      'YDYE': 'YEM',
                      'DDR': 'DEU'}
    df_speech['ISO-alpha3 Code'].replace(speech_replace, inplace=True)

    # prepare df_happy key to merge with df_code
    # Taiwan Provice of China not in df_code
    happy_replace = {'Bolivia': 'Bolivia (Plurinational State of)',
                     'Congo (Brazzaville)': 'Congo',
                     'Congo (Kinshasa)': 'Democratic Republic of the Congo',
                     'Czech Republic': 'Czechia',
                     'Hong Kong S.A.R. of China': 'China - Hong Kong Special Administrative Region',
                     'Iran': 'Iran (Islamic Republic of)',
                     'Ivory Coast': 'Cote d’Ivoire',
                     'Laos': "Lao People's Democratic Republic",
                     'Moldova': 'Republic of Moldova',
                     'North Cyprus': 'Cyprus',
                     'Palestinian Territories': 'State of Palestine',
                     'Russia': 'Russian Federation',
                     'Somaliland region': 'Somalia',
                     'South Korea': 'Republic of Korea',
                     'Swaziland': 'Eswatini',
                     'Syria': 'Syrian Arab Republic',
                     'Tanzania': 'United Republic of Tanzania',
                     'United Kingdom': 'United Kingdom of Great Britain and Northern Ireland',
                     'United States': 'United States of America',
                     'Venezuela': 'Venezuela (Bolivarian Republic of)'}
    df_happy.replace(happy_replace, inplace=True)

    # prepare df_hdi key to merge with df_code
    hdi_replace = {'Congo (Democratic Republic of the)': 'Democratic Republic of the Congo',
                   "Côte d'Ivoire": "Cote d'Ivoire",
                   'Eswatini (Kingdom of)': 'Eswatini',
                   'Hong Kong - China (SAR)': 'China - Hong Kong Special Administrative Region',
                   'Korea (Republic of)': 'Republic of Korea',
                   'Moldova (Republic of)': 'Republic of Moldova',
                   'Tanzania (United Republic of)': 'United Republic of Tanzania',
                   'United Kingdom': 'United Kingdom of Great Britain and Northern Ireland',
                   'United States': 'United States of America',
                   'Viet Nam': 'Vietnam'}
    df_hdi.replace(hdi_replace, inplace=True)
    df_hdi['hdi_class'] = df_hdi['hdi_score'].apply(hdi_classification)  # classification label

    # === MERGE DATASETS ===
    df = pd.merge(df_code, df_speech,
                  how='right', on='ISO-alpha3 Code')  # add speech

    df = pd.merge(df, df_happy,
                  how='left',
                  left_on=['Country or Area', 'Year'],
                  right_on=['Country name', 'Year'])  # add happiness metrics
    df.drop('Country name', axis=1, inplace=True)

    df = pd.merge(df, df_hdi,
                  how='left',
                  left_on=['Country or Area', 'Year'],
                  right_on=['Country', 'Year'])  # add hdi score
    df.drop('Country', axis=1, inplace=True)

    return df
