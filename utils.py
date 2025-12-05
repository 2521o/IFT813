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
        X = df.iloc[:, 0].values  # East
        Y = df.iloc[:, 1].values  # North
        pH = df.iloc[:, 4].values
        return X, Y, pH

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


def modele_spherique(h, nugget, sill, range_val):
    """
    Modèle Sphérique
    """
    # CORRECTION : On s'assure que h est un array mais on garde sa forme (2D ou 1D)
    h_arr = np.asarray(h)
    val = np.zeros_like(h_arr)

    # On évite la division par zéro
    if range_val < 1e-9:
        range_val = 1e-9

    mask = h_arr <= range_val
    # Formule du cours appliquée élément par élément
    val[mask] = nugget + (sill - nugget) * (
        1.5 * (h_arr[mask] / range_val) - 0.5 * (h_arr[mask] / range_val) ** 3
    )
    val[~mask] = sill

    # Correction mathématique à l'origine (gamma(0)=0)
    # Mais le nugget s'applique dès h > 0
    val[h_arr == 0] = 0

    return val  # On renvoie la matrice telle quelle (pas de flatten!)


def modele_exponentiel(h, nugget, sill, range_val):
    """
    Modèle Exponentiel
    """
    h_arr = np.asarray(h)
    val = nugget + (sill - nugget) * (1 - np.exp(-3 * h_arr / range_val))
    val[h_arr == 0] = 0
    return val
