import os
import requests
import pandas as pd


def download_covid_confirmed_data(url, local_path):
    """
    Download the COVID-19 confirmed cases dataset if it doesn't already exist.

    Parameters:
        url (str): The URL to download the CSV file from.
        local_path (str): The local file path where the CSV should be saved.
    """
    if not os.path.exists(local_path):
        print(f"Downloading data from {url} ...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(local_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded data to {local_path}")
        else:
            raise Exception(f"Failed to download data. Status code: {response.status_code}")
    else:
        print(f"Data file {local_path} already exists.")


def load_covid_confirmed_data(local_path):
    """
    Load and preprocess the COVID-19 confirmed cases dataset from a local CSV file.

    Parameters:
        local_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The data in long format with a datetime column.
    """
    df = pd.read_csv(local_path)
    # Transform the wide-format data (dates as columns) to a long format.
    df_long = df.melt(
        id_vars=["Province/State", "Country/Region", "Lat", "Long"],
        var_name="Date",
        value_name="Confirmed"
    )
    df_long["Date"] = pd.to_datetime(df_long["Date"])
    return df_long


def download_and_load_covid_confirmed_data():
    """
    Download the COVID-19 confirmed cases dataset and load it into a DataFrame.

    Returns:
        pd.DataFrame: The preprocessed data.
    """
    url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/archived_data/archived_time_series/time_series_19-covid-Confirmed_archived_0325.csv"
    local_path = "data/time_series_19-covid-Confirmed_archived_0325.csv"
    # Ensure the data directory exists.
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    download_covid_confirmed_data(url, local_path)
    df_long = load_covid_confirmed_data(local_path)
    return df_long
