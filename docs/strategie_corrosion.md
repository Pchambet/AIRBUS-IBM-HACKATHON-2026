# Livre Blanc : Architecture Systémique et Mathématique pour la MRO (Predictive Maintenance)

## Vision Produit : Du Classifieur au Moteur de Risque
L'objectif de la maintenance aéronautique prédictive (MRO) n'est pas de deviner une "date exacte" d'apparition de corrosion (ce qui est stochastiquement impossible en raison de variables latentes comme la qualité des traitements initiaux), mais de fournir au planificateur une **probabilité de risque d'état fiable à 100% à un instant $T$ imposé**.

Un score de "75% de risque" permettra au moteur de décision (le Produit) de calculer l'**Espérance de Coût** : 
$0.75 \times \text{Coût d'immobilisation non planifiée}$. Si cette espérance dépasse le coût d'une inspection avancée, la décision métier d'avancer la *C-Check* devient automatique et justifiée.

C'est pourquoi l'évaluation algorithmique ne se fait pas sur l'AUC, mais sur le **Brier Score**.

## 1. L'Axe Mathématique : Décomposition du Brier Score

Le Brier Score mesure la distance quadratique moyenne entre la probabilité prédite $P_i$ et l'événement binaire $Y_i$ au moment de l'inspection (Snapshot stochastique).
**$\text{Brier Score} = \text{Fiabilité (Calibration)} - \text{Résolution} + \text{Incertitude}$**

### Optimisation de la Résolution (Feature Engineering Expert)
Pour maximiser la capacité du modèle à séparer les avions sains des avions à risque, nous calculons l'aire sous la courbe des contraintes environnementales *jusqu'à* la date d'inspection.
1. **L'Effet "Cocotte-Minute" Stabilisé** : La corrosion s'accélère via l'interaction `Temps de Parking * Humidité * Sels Marins Combinés`. Pour éviter l'explosion de la variance due à la multiplication de variables (création d'outliers massifs qui induisent l'overfitting), nous appliquons une transformation stricte : $\log(1 + x)$.
2. **Le Facteur de Décroissance (Decay Factor)** : L'utilisation de sommes cumulées infinies (`cumsum`) ignore la maintenance courante (lavages, pluie). Nous utilisons donc une **Moyenne Mobile Pondérée Exponentiellement (EWMA)**. L'exposition d'il y a 3 mois compte davantage que celle d'il y a 4 ans.
3. **Mémoire Climatique (Moyennes Longues)** : Ajout de moyennes glissantes sur 12 et 24 mois pour l'humidité et la température.
4. **Indicateur de Risque Acide** : Création de la variable d'interaction experte $(SO_2 + NO_2) \times Humidit\acute{e}$.

### Optimisation de la Fiabilité (La Calibration)
Les algorithmes d'arbres boostés (LightGBM) poussent naturellement les probabilités vers les extrêmes et sont très mal calibrés.
- **La Sécurité "Platt Scaling" :** Avec seulement ~1% de cas positifs, une Calibration Isotonique (non-paramétrique) risque de sur-apprendre le bruit des données de validation. Nous utilisons systématiquement le **Platt Scaling** (Méthode Sigmoïde), qui offre une correction paramétrique robuste et lisse, garantissant que nos probabilités sont mathématiquement "vraies".

## 2. L'Axe de Validation : Grouped Time-Series Split

La faille majeure des modèles MRO est le *Data Leakage*. Entraîner un modèle sur des données de 2021 et 2020 avec un `train_test_split` classique permet au modèle d'apprendre l'identité d'un avion ou de voir le futur.

**Notre Pare-Feu Temporel Absolu :**
Nous implémentons une Validation Croisée sur-mesure (Grouped Time-Series Split) en deux étapes :
1. **Protection d'Identité (K-Fold sur `aircraft_id`)** : L'avion présent dans le pli de validation n'a *jamais* été vu (même à une date antérieure) dans les données d'entraînement du modèle.
2. **Protection Chronologique** : Lors de l'évaluation sur le pli de validation (ex: inspection de l'avion $X$ le 14 Octobre 2022), toutes les données d'entraînement des autres avions postérieures à cette date stricte sont effacées du set d'apprentissage.

Cette architecture blindée garantit un score interne (OOF - Out of Fold) parfaitement corrélé avec la performance réelle d'évaluation (Leaderboard).
