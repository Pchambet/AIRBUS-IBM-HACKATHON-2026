# Airbus × IBM × AWS 2026 — Prédiction de corrosion aéronautique

**Résultats : 6ᵉ sur le leaderboard Kaggle (métrique Brier) · 2ᵉ au classement général du
hackathon (modèle + pitch business).**

Le modèle est remonté de la **22ᵉ place publique à la 6ᵉ en privé** : calibré pour la
robustesse, il a tenu sur le Private LB pendant que les équipes sur-ajustant le public
s'effondraient. Un pitch business solide de l'équipe a ensuite porté le projet à la **2ᵉ place
générale**. *(Modèle conçu par Pierre Chambet ; projet d'équipe.)*

Prédire le risque de corrosion mensuel d'un avion à partir de son historique environnemental
(météo METAR, aérosols, gaz, temps de stationnement).

## Le problème, correctement posé

Le score n'est calculé que sur **2 points par avion de test** : le mois de corrosion **T**
(vraie valeur = 1) et le mois **24 mois plus tôt** T-24 (= 0). C'est donc un problème de
**discrimination intra-avion, parfaitement balancé 50/50**.

Conséquence clé : **tout ce qui est constant par avion** (type, route habituelle,
susceptibilité propre) **s'annule** entre T et T-24. Seul le **changement temporel
intra-avion** (exposition accumulée, âge, dose de corrosion) est exploitable.

## L'approche gagnante

1. **Cible** : T = 1, T-24 = 0 (set balancé).
2. **Features** : historique causal par avion — EWMA des aérosols, moyennes glissantes
   météo, dose cumulée / âge / mois observés.
3. **Modèle** : LightGBM `objective='regression'` (MSE = Brier exact).
4. **Le levier décisif — shrinkage α=0.7** : les avions train et test sont disjoints avec un
   fort décalage de distribution (AUC adversariale ≈ 0.82). Le modèle brut est *surconfiant*
   hors distribution. Rétrécir les prédictions vers 0.5 (`p → 0.5 + 0.7·(p−0.5)`) corrige
   cette surconfiance. **α=0.7 a été réglé sur une validation OOD qui reproduisait le vrai
   score public au millième (0.218 estimé vs 0.22 observé).**

Ce n'est ni un modèle exotique ni une feature magique qui a gagné, mais une **calibration
robuste guidée par une validation honnête** — d'où la remontée du public (0.215) au privé (0.18).

## Reproduire

```bash
pip install -r requirements.txt
python train_final_model.py        # -> data/final_submission_best.csv
```

## Structure

```
train_final_model.py   Pipeline gagnant (entraîne + génère la soumission)
data/                  Données d'entrée + final_submission_best.csv (soumission gagnante)
analysis/              La rigueur qui a gagné :
  adversarial.py         prouve le décalage train/test (AUC 0.82)
  robust_ood.py          validation OOD qui reproduit le vrai leaderboard
  leak_audit.py          chasse au leak (négative -> aucun exploit)
  objective_test.py      MSE vs binary (réfute les sur-promesses "0.10")
  lb_noise.py            significativité statistique du leaderboard (143 obs)
eda/                   Analyse exploratoire (script, rapport, figures)
docs/                  Notes stratégiques et structure du pitch
```

## Enseignements

- Le leaderboard **public** ne portait que sur ~143 observations : erreur-type ≈ 0.013–0.016,
  donc tout écart < ~0.02–0.03 est du **bruit**. Les rangs 4–22 du public étaient un match nul
  statistique ; seul le **privé (99 %)** était décisif.
- Aucun leak exploitable n'existe dans les features (audit complet). Les scores publics
  extrêmes (≈0.04) relevaient du *leaderboard probing* sur l'échantillon public minuscule.
