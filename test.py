import matplotlib.pyplot as plt
import numpy as np
from model import ModeleAR_MAP
from utils import load_transform

dates, prix_orig, data_stat = load_transform("data/IBM.txt")


# On teste les ordres p de 1 à 25
resultats_aic = []
erreurs_mse = []
plage_p = range(1, 26)
lambda_reg = 0.1

for p in plage_p:
    model = ModeleAR_MAP(p=p, lambda_reg=lambda_reg)
    model.fit(data_stat)
    # resultats_aic.append(model.aic)
    erreurs_mse.append(model.sigma_e)

# Le meilleur ordre est celui qui minimise l'erreur MSE
best_p = np.argmin(erreurs_mse) + 1
print()
print(f"\nMeilleur ordre trouvé : p = {best_p}")
print(f"Variance de l'erreur (MSE) : {erreurs_mse[best_p-1]:.2f}")

# Estimation du modèle final
final_model = ModeleAR_MAP(p=best_p, lambda_reg=lambda_reg)
final_model.fit(data_stat)

# Synthèse de nouvelles données
data_synth = final_model.synthetiser(data_stat, len(data_stat))

# Visualisations
plt.figure(figsize=(14, 10))

# Données d'origine
plt.subplot(2, 2, 1)
plt.plot(dates, prix_orig, color="black", label="Prix IBM")
plt.title("Données d'origine (Prix non stationnaires)")
plt.legend()
plt.grid(True)

# Données stationnaires vs Synthèse
plt.subplot(2, 2, 2)
plt.plot(data_stat, color="blue", alpha=0.6, label="Log-Rendements Réels")
plt.plot(data_synth, color="orange", alpha=0.6, label=f"Synthèse AR({best_p})")
plt.title(f"Données transformées vs Synthèse MAP")
plt.legend()
plt.grid(True)

# Histogramme de l'erreur
X_mat, Y_vec = final_model.compute_matrix(data_stat)
pred_finale = X_mat @ final_model.coeffs
residus_finaux = Y_vec - pred_finale

plt.subplot(2, 2, 3)
plt.hist(
    residus_finaux,
    bins=50,
    color="green",
    density=True,
    alpha=0.7,
    label="Histogramme Erreur",
)
# Ajout courbe normale théorique pour vérifier l'hypothèse Gaussienne
x_axis = np.linspace(min(residus_finaux), max(residus_finaux), 100)
pdf_theorique = (1 / np.sqrt(2 * np.pi * final_model.sigma_e)) * np.exp(
    -0.5 * x_axis**2 / final_model.sigma_e
)
plt.plot(x_axis, pdf_theorique, "r-", linewidth=2, label="Normale Théorique")
plt.title("Histogramme de l'erreur")
plt.legend()
plt.grid(True)

# Graphe 4 : Évolution de l'erreur (Choix de l'ordre)
plt.subplot(2, 2, 4)
plt.plot(plage_p, erreurs_mse, marker="o", color="purple")
plt.axvline(x=best_p, color="red", linestyle="--", label=f"Best p={best_p}")
plt.title("Évolution de la Variance de l'erreur selon p")
plt.xlabel("Ordre p")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Affichage des coefficients pour le rapport
print("\nCoefficients du modèle AR optimal (a0, a1, ...):")
print(final_model.coeffs.flatten())
