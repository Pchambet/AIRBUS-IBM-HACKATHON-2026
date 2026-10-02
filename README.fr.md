# AIRBUS-IBM-HACKATHON-2026 (version française)

*English version (référence, plus détaillée) : [README.md](README.md)*

Modèle de risque de corrosion pour le hackathon Airbus × IBM × AWS 2026, et la validation
« hors distribution » qui a fixé son unique paramètre décisif.

**Résultat (tel que publié par les organisateurs) :** 6ᵉ au leaderboard Kaggle (score de Brier) et
2ᵉ au classement général (modèle + pitch business). Le modèle est passé de la 22ᵉ place publique
(Brier 0,215) à la 6ᵉ place privée (0,18). Projet d'équipe ; j'étais responsable ML et j'ai construit
la chaîne de modélisation et de validation de ce dépôt. Le pitch business était un travail d'équipe.

![Le shrinkage optimal diffère entre validation croisée et avions proches du test](docs/figures/hero_shrinkage.png)

## L'essentiel

- **La flotte de test est une autre population.** Un classifieur adversarial (plis groupés par
  avion) distingue les lignes de test des lignes d'entraînement avec une AUC de **0,86** ; 63 % des
  avions de test ont des données dès la première année de l'historique, contre 0,3 % à l'entraînement.
- **La validation croisée groupée est optimiste.** Brier du modèle brut : **0,140** en distribution,
  **0,204** sur les 315 avions d'entraînement les plus proches du test.
- **Un seul paramètre compte.** Le rétrécissement `p → 0,5 + α (p − 0,5)` ne sert à rien en
  distribution (α optimal = 1,00) mais est optimal à **α = 0,72** sur les avions proches du test.
  L'α = 0,7 soumis améliore le Brier de cet échantillon de **0,008** (IC 95 % apparié 0,002–0,014),
  à **0,197**.
- **L'essentiel du signal transférable est l'horloge.** Deux variables temporelles seules
  récupèrent **86 %** du gain du modèle complet (66 variables) face à une prévision constante.
- **Le leaderboard public était du bruit.** Sur 143 lignes, un score de Brier a une marge d'erreur
  de ±**0,031** ; deux équipes doivent différer de plus de **0,043** pour être départagées.

## Reproduire

```bash
make setup && make data && make run && make report
```

Les données du concours ne sont pas redistribuables : voir [`data/README.md`](data/README.md).
La démarche complète (en anglais) est dans [`docs/approach.md`](docs/approach.md) et dans le
[rapport interactif](https://pchambet.github.io/AIRBUS-IBM-HACKATHON-2026/).

---

Réalisé par [Pierre Chambet](https://github.com/Pchambet) — science de la décision pour des opérations sous incertitude.
