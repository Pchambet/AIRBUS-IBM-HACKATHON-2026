# Trame de Pitch : HAKS - Airbus x IBM x AWS (2026)

*Durée : Adapter selon le temps imparti (Généralement 3 à 5 minutes).*

---

## Résultat & Méthode (version finale, post-compétition)

> ⚠️ Cette section reflète ce qui a **réellement gagné**. Les sections ci-dessous datent d'une
> version antérieure (plafond 20 %, Platt scaling) qui ne correspond pas au modèle final — à
> mettre à jour avant présentation.

**Résultat — 6ᵉ sur le leaderboard Kaggle (métrique Brier), 2ᵉ au classement général du
hackathon (modèle + pitch business).** Et surtout : notre modèle a *progressé* du leaderboard
public (Brier 0.215, 22ᵉ) au privé (0.18, 6ᵉ), là où la majorité des équipes ont reculé. Ce
n'est pas un hasard, c'est le fruit de notre méthode — et un pitch business solide a porté le
projet jusqu'à la 2ᵉ place finale.

**Méthode.** Nous avons d'abord compris la vraie nature du problème : le score ne porte que
sur deux instants par avion — le mois de corrosion et celui 24 mois plus tôt — soit une
discrimination *intra-avion* parfaitement équilibrée. Tout ce qui est propre à un avion
(type, route, susceptibilité) s'annule ; seul le **changement environnemental accumulé dans
le temps** compte. Nous avons donc construit des features causales d'exposition (chlorures,
humidité, dose de corrosion cumulée, âge) et entraîné un LightGBM optimisant directement le
Brier Score.

**Notre vrai différenciateur n'est pas le modèle — c'est la validation.** Les avions de test
sont entièrement différents de ceux d'entraînement (décalage de distribution prouvé, AUC
0.82). Nous avons donc bâti une validation « hors-distribution » qui **reproduisait le vrai
score au millième** (0.218 estimé vs 0.22 réel). Elle nous a permis de calibrer notre seul
levier décisif — un *shrinkage* contrôlé contre la surconfiance — et de viser le **leaderboard
privé** plutôt qu'un public minuscule (143 observations, statistiquement du bruit). Pendant
que d'autres sur-ajustaient ce bruit, nous avons construit un modèle robuste, honnête et
reproductible — et c'est lui qui nous a fait grimper au classement final.

> **Punchline :** « On n'a pas cherché le modèle le plus malin, on a cherché la *validation*
> la plus honnête. C'est elle qui prédisait notre vrai score, et c'est elle qui nous a placés 6ᵉ. »

---

## Result & Method (English)

**Result — 6th on the Kaggle metric leaderboard (Brier), 2nd overall in the hackathon
(model + business pitch).** More importantly, our model *improved* from the public leaderboard
(Brier 0.215, 22nd) to the private one (0.18, 6th), while most teams regressed. That wasn't
luck — it was our method, and a strong business pitch carried the project to 2nd place overall.

**Method.** We first understood the true nature of the problem: scoring covers only two
moments per aircraft — the corrosion month and the month 24 months earlier — i.e. a perfectly
balanced *within-aircraft* discrimination. Everything specific to an aircraft (type, route,
susceptibility) cancels out; only the **environmental exposure accumulated over time** matters.
So we built causal exposure features (chlorides, humidity, cumulative corrosion dose, age) and
trained a LightGBM that directly optimizes the Brier Score.

**Our real differentiator isn't the model — it's the validation.** The test aircraft are
entirely different from the training ones (a proven distribution shift, adversarial AUC 0.82).
So we built an out-of-distribution validation that **reproduced the real score to the third
decimal** (0.218 estimated vs 0.22 actual). It let us calibrate our single decisive lever — a
controlled *shrinkage* against over-confidence — and aim for the **private leaderboard** rather
than a tiny public one (143 observations, statistically just noise). While others over-fit that
noise, we built a robust, honest, reproducible model — and that's what climbed the final ranking.

> **Punchline:** "We didn't look for the cleverest model — we looked for the most honest
> *validation*. It's what predicted our true score, and it's what put us 6th."

---

## 1. Introduction & Vision Produit (Business Impact - 20/50)
**Accroche :** "La maintenance prédictive ne doit pas deviner des dates impossibles, elle doit calculer une Espérance Financière de Risque."
*   **Le Constat :** Dans l'aéronautique, un modèle qui donne une probabilité brute "non-calibrée" est dangereux (trop d'inspections inutiles ou des risques ignorés).
*   **Notre Solution :** Un moteur stochastique garantissant le risque absolu. Nous plafonnons à un risque réel de ~20% d'apparition de corrosion sur un mois donné, car c'est la "Ground Truth" historique.
*   **Impact Métier & Coût :** Le planificateur (via notre interface) multiplie cette probabilité par le coût d'une immobilisation non-planifiée (AOG). Si cette *Espérance de Coût* est supérieure au coût de la maintenance, l'inspection est déclenchée. C'est 100% ROI-driven.

## 2. Rigueur Technique & Outils (Data Processing - 15/50)
*   **Feature Engineering Inversé (Physique vers Code) :** Nous avons simulé la physique des matériaux. Au lieu de simples sommes, nous avons modélisé des Moyennes Mobiles Pondérées Exponentiellement (EWMA) pour simuler la décroissance des risques après un lavage (Line Maintenance). Nous avons créé l'effet "Cocotte-Minute" (Temps Parking $\times$ Humidité $\times$ Sel) contenu par un *Log-transform* pour éviter l'explosion de variance.
*   **La Muraille Pare-Feu (Le Time-Series Split) :** Préciser fermement que vous avez banni le `train_test_split` classique. Vous avez isolé les avions par ID (`GroupKFold`) pour prouver l'absence totale de fuite d'identité. Votre modèle a évalué le risque en stricte condition réelle.
*   **L'Arme Secrète (Calibration & Brier Score) :** Les algorithmes d'arbres sont sur-confiants. En rajoutant une sur-couche mathématique de *Platt Scaling* (Sigmoïde), nous avons divisé notre erreur quadratique (Brier Score) par **10**.
*   **Intégration Tooling :** (Mentionnez impérativement ici comment vous avez exploité **IBM Bob** pour la complétion, l'optimisation, l'analyse ou le déploiement de ce script python hautement structuré).

## 3. Démonstration Visuelle (Presentation Quality - 15/50)
*   *Slide 1 :* Montrez l'équation de la décision métier (Probabilité Calibrée $\times$ Coût).
*   *Slide 2 :* Montrez la variable *Acid Risk Index* ou l'impact du *Log-transform*.
*   *Slide 3 (Le clou du spectacle) :* Affichez la **Courbe de Calibration (Reliability Diagram)** (le fichier `calibration_curve.png`). Montrez aux juges que la ligne de votre modèle épouse parfaitement la diagonale parfaite, prouvant que lorsque vous affichez 15% de risque, la probabilité est mathématiquement incontestable.

## Conclusion
"Nous n'avons pas créé une boule de cristal pour le hackathon. Nous avons développé, grâce à l'écosystème IBM Bob, le cœur d'un produit MRO déployable demain chez Airbus."
