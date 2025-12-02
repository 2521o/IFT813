import pandas as pd
import numpy as np


def load_ibm_data(filename: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filename, sep=",")
        df.columns = df.columns.str.strip()
        df.Date = pd.to_datetime(df.Date)
        return df

    except Exception as e:
        print(f"Erreur lors de la lecture du fichier : {e}")
        return None


def load_soil_data(filename: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filename, sep="\s+", skiprows=4, header=None)
        df.columns = ["East", "North", "K", "logK", "pH", "P", "logP"]
        return df

    except Exception as e:
        print(f"Erreur : {e}")
        return None


def load_transform(filename):
    """
    Charge les données et applique la transformation pour rendre la série stationnaire.
    """
    # Lecture du format spécifique Date,High
    df = load_ibm_data(filename)
    dates = df.Date.values
    prix = df.High.values

    # Transformation logarithmique
    log_prix = np.log(prix)
    # On retire la tendance
    data_stat = np.diff(log_prix)

    return dates, prix, data_stat
