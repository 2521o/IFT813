import numpy as np
import matplotlib.pyplot as plt
from utils import model_gaussien, model_spherique
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit


class ModeleAR:
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


class KrigingModel:
    def __init__(self, type_modele="spherical"):
        self.func = (
            modele_spherique if type_modele == "spherical" else modele_exponentiel
        )
        self.params = None  # [Nugget, Sill, Range]
        self.coeffs_trend = None
        self.X_train = None
        self.Y_train = None
        self.pH_res = None  # Résidus

    def fit(self, X, Y, pH):
        """
        Tendance + Variogramme + Paramètres
        """
        self.X_train, self.Y_train = X, Y
        N = len(X)

        # On suppose que le pH dépend de la position pH = a*X + b*Y + c
        matrice_reg = np.column_stack((X, Y, np.ones(N)))  # [X, Y, 1]

        # coeffs = (A.T * A)^-1 * A.T * pH
        self.coeffs_trend = (
            np.linalg.inv(matrice_reg.T @ matrice_reg) @ matrice_reg.T @ pH
        )
        # On calcule les résidus
        trend = matrice_reg @ self.coeffs_trend
        self.pH_res = pH - trend  # Données stationnaires

        # Calcul des distances entre tous les points
        self.coords = np.column_stack((X, Y))
        dists = pdist(self.coords)

        # Calcul des différences de pH au carré
        pH_diff = pdist(self.pH_res.reshape(-1, 1), metric="sqeuclidean")
        # Moyenne par tranches de distance
        bins = np.linspace(0, np.max(dists) / 2, 20)  # On s'arrête à D/2
        lags = []
        gammas = []

        # Discretisation h
        for i in range(len(bins) - 1):
            # On prend toutes les paires qui sont dans cet intervalle de distance
            mask = (dists >= bins[i]) & (dists < bins[i + 1])
            if np.sum(mask) > 0:
                lags.append(np.mean(dists[mask]))  # Distance moyenne
                gammas.append(0.5 * np.mean(pH_diff[mask]))  # Variogramme

        # On cherche nugget, sill, range qui collent aux points empiriques
        p0 = [0, np.var(self.pH_res), np.mean(lags)]  # Estimation initiale
        self.params, _ = curve_fit(self.func, lags, gammas, p0=p0, bounds=(0, np.inf))

        return lags, gammas

    def predict(self, x_cible, y_cible):
        """
        Prédit le pH à une nouvelle position (x, y)
        """
        N = len(self.X_train)

        # On reconstruit la matrice des distances
        dist_mat = squareform(pdist(self.coords))
        K = self.func(dist_mat, *self.params)  # On applique le modèle (Sphérique/Exp)

        # Ajout de la ligne/colonne de 1 (Contrainte Lagrange)
        K_aug = np.ones((N + 1, N + 1))
        K_aug[:N, :N] = K
        K_aug[N, N] = 0

        # Relations vers la cible
        dist = np.sqrt((self.X_train - x_cible) ** 2 + (self.Y_train - y_cible) ** 2)
        D = np.ones(N + 1)
        D[:N] = self.func(dist, *self.params)

        # Résolution du système
        poids_aug = np.linalg.inv(K_aug) @ D

        w = poids_aug[:N]  # Poids du krigeage

        # On prédit le résidu
        pred_residu = np.sum(w * self.pH_res)
        # On rajoute la tendance à cet endroit
        pred_trend = (
            self.coeffs_trend[0] * x_cible
            + self.coeffs_trend[1] * y_cible
            + self.coeffs_trend[2]
        )

        # Variance d'erreur
        var_err = np.sum(poids_aug * D)

        return pred_residu + pred_trend, var_err
