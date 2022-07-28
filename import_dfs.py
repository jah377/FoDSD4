import os
import re
import numpy as np
import pandas as pd
from dataprep.clean import clean_country


def hdi_classifier(score: float):
    """ 
    convert hdi score to classification as defined by http://hdr.undp.org/
    """

    if score < 0.550:
        hdi_class = 'Low'
    elif score < 0.7:
        hdi_class = 'Medium'
    elif score < 0.8:
        hdi_class = 'High'
    elif score >= 0.8:
        hdi_class = 'Very High'
    else:
        hdi_class = np.nan

    return hdi_class


def import_merge():
    """
    import relevant files, process primary keys, and merge datasets
    """

    # === IMPORT SPEECHES ===
    sessions = np.arange(25, 76)
    data = []

    for session in sessions:
        directory = "data/speeches/Session " + \
            str(session)+" - "+str(1945+session)

        for filename in os.listdir(directory):
            f = open(os.path.join(directory, filename))

            if filename[0] == ".":  # ignore hidden files
                continue
            splt = filename.split("_")
            data.append([session, 1945+session, splt[0], f.read()])

    df_speech = pd.DataFrame(
        data, columns=['Session', 'Year', 'ISO-alpha3 Code', 'Speech'])

    # === IMPORT ACCESSORY DATASETS ===
    df_code = pd.read_csv('data/country_codes.csv')

    hdi_data = pd.read_csv('data/human_development_index.csv', sep=';')
    df_hdi = pd.melt(hdi_data,
                     id_vars=['Country'],
                     value_vars=hdi_data.columns[2:])
    df_hdi.columns = ['Country', 'Year', 'hdi_score']
    df_hdi['Year'] = df_hdi['Year'].apply(int)
    df_hdi['Country'] = df_hdi['Country'].str.strip()
    df_hdi['hdi_score'].replace('..', np.nan, inplace=True)
    df_hdi['hdi_score'] = df_hdi['hdi_score'].apply(float)

    # === CLEAN UP KEYS PRIOR TO MERGE ===
    # use df_code as primary key -- inspect for weird characters
    to_replace = [s for s in df_code['Country or Area']
                  if re.search(r'[À-ÿ]+', s)]
    new_countries = ['Reunion', "Cote d'Ivoire",
                     'Curacao', 'Saint Barthelemy', 'Aland Island']
    replace_dict = dict(zip(to_replace, new_countries))
    df_code['Country or Area'].replace(replace_dict, inplace=True)
    df_code['Country or Area'].replace({'Viet Nam': 'Vietnam'}, inplace=True)

    # prepare df_speech key to merge with df_code
    # EU -- unsure what to do
    # CSK -- Czechoslovakia divided into Czechia (CZE), and Slovakia (SVK)
    # YUG -- divided into multiple countries
    df_speech['ISO-alpha3 Code'].replace({'POR': 'PRT',
                                          'YDYE': 'YEM',
                                          'DDR': 'DEU'}, inplace=True)

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
    df_hdi['hdi_class'] = df_hdi['hdi_score'].apply(
        hdi_classifier)  # classification label

    # === MERGE DATASETS ===
    df = pd.merge(df_code, df_speech,
                  how='right', on='ISO-alpha3 Code')  # add speech

    df = pd.merge(df, df_hdi,
                  how='left',
                  left_on=['Country or Area', 'Year'],
                  right_on=['Country', 'Year'])  # add hdi score
    df.drop('Country', axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def clean_dataframe(df):
    """
    clean non-string data
    """

    # Impute null 'Country or Area' and 'Region Name'
    yug_idx = df[df['ISO-alpha3 Code'] == 'YUG'].index.tolist()
    csk_idx = df[df['ISO-alpha3 Code'] == 'CSK'].index.tolist()
    eu_idx = df[df['ISO-alpha3 Code'] == 'EU'].index.tolist()

    df.at[yug_idx, 'Country or Area'] = ['Yugoslavia']*len(yug_idx)
    df.at[csk_idx, 'Country or Area'] = ['Czechoslovakia']*len(csk_idx)
    df.at[eu_idx, 'Country or Area'] = ['European Union']*len(eu_idx)

    df.at[yug_idx, 'Region Name'] = ['Europe']*len(yug_idx)
    df.at[csk_idx, 'Region Name'] = ['Europe']*len(csk_idx)
    df.at[eu_idx, 'Region Name'] = ['Europe']*len(eu_idx)

    # Impute null 'Sub-region Name' -- only CSK and YUG
    df['Sub-region Name'].fillna('Eastern Europe', inplace=True)  # CSK and YUG

    # Clean up 'Country or Area' name
    cleaned = clean_country(df[['Country or Area']],
                            'Country or Area',
                            output_format='name')

    df['country_cleaned'] = cleaned['Country or Area_clean']
    df['country_cleaned'] = df.apply(lambda x: (x['Country or Area']
                                                if pd.isna(x['country_cleaned'])
                                                else x['country_cleaned']),
                                     axis=1)

    # drop unnecessary columns
    delete_cols = ['Global Code', 'Global Name', 'Region Code',
                   'Sub-region Code', 'Intermediate Region Code',
                   'M49 Code', 'ISO-alpha2 Code',
                   'Least Developed Countries (LDC)',
                   'Land Locked Developing Countries (LLDC)',
                   'Small Island Developing States (SIDS)',
                   'Developed / Developing Countries']
    df.drop(delete_cols, axis=1, errors='ignore', inplace=True)

    return df
