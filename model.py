import numpy as np
import matplotlib.pyplot as plt
from utils import load_ibm_data, load_transform


class ModeleAR_MAP:
    def __init__(self, p, lambda_reg=0.1):
        """
        p : Ordre du modèle AR
        lambda_reg : Paramètre de régularisation
        """
        self.p = p
        self.lambda_reg = lambda_reg
        self.coeffs = None  # Vecteur a
        self.sigma_e = None  # Variance du bruit estimée empiriquement
        self.aic = None  # Critère pour le choix du modèle

    def compute_matrix(self, data):
        """
        Construit les matrices X et Y pour la regression
        Y : Vecteur colonne des valeurs à prédire
        X : Matrice où chaque ligne contient [1, y_{t-1}, ..., y_{t-p}]
        """
        N = len(data)
        X = []
        Y = []

        # On commence à p car on a besoin de p valeurs passées
        for t in range(self.p, N):
            # Le vecteur Y contient la valeur cible à l'instant t
            Y.append(data[t])

            row = [1.0]  # Biais a0
            for lag in range(1, self.p + 1):
                row.append(data[t - lag])  # Y_t-tau pour tau=1..p
            X.append(row)

        return np.array(X), np.array(Y)

    def fit(self, data):
        """
        Estimation MAP des coefficients.
        Formule : a_hat = (X.T*X + lambda*I)^-1 * X.T * Y
        """
        X, Y = self.compute_matrix(data)

        # Calcul de la matrice de corrélation
        XTX = X.T @ X

        # On ajoute lambda sur la diagonale
        reg_matrix = self.lambda_reg * np.eye(XTX.shape[0])
        reg_matrix[0, 0] = 0

        self.coeffs = np.linalg.inv(XTX + reg_matrix) @ X.T @ Y

        # Calcul de l'erreur pour estimer Sigma_e
        predictions = X @ self.coeffs
        residus = Y - predictions
        self.sigma_e = np.var(residus)  # Variance empirique de l'erreur

    def synthetiser(self, init_data, n_steps):
        """
        Génère de nouvelles données artificielles.
        """
        history = list(init_data[-self.p :])  # On amorce avec la fin des vraies données
        synthese = []

        for _ in range(n_steps):
            val = self.coeffs[0]
            # Boucle a0 + a1*y_t-1 + ... + ap*y_t-p
            for i in range(1, self.p + 1):
                val += self.coeffs[i] * history[-i]

            # Ajout du bruit gaussien
            bruit = np.random.normal(0, np.sqrt(self.sigma_e))

            nouvelle_val = val + bruit
            synthese.append(nouvelle_val)
            history.append(nouvelle_val)  # Mise à jour de l'historique

        return np.array(synthese)


#
