# AIRBUS-IBM-HACKATHON-2026 (version française)

*English version (référence, plus détaillée) : [README.md](README.md)*

Modèle de risque de corrosion pour le hackathon Airbus × IBM × AWS 2026, et la validation
« hors distribution » qui a fixé son unique paramètre décisif.

**Résultat (tel que publié par les organisateurs) :** 6ᵉ au leaderboard Kaggle (score de Brier) et
2ᵉ au classement général (modèle + pitch business). Le modèle est passé de la 22ᵉ place publique
(Brier 0,215) à la 6ᵉ place privée (0,18) ; les deux classements portent sur peu de lignes, cette
remontée est donc à lire avec les limites ci-dessous. Projet d'équipe : le code écrit pendant
l'événement est dans le dépôt de l'équipe, [Yixian-ch/AIRBUS-IBM](https://github.com/Yixian-ch/AIRBUS-IBM).
J'étais responsable ML et je portais la chaîne de prédiction du risque de corrosion ; ce dépôt en est
la reconstruction et la ré-analyse après l'événement. Le pitch business était un travail d'équipe.

![Le shrinkage optimal diffère entre validation croisée et avions proches du test](docs/figures/hero_shrinkage.png)

## L'essentiel

- **La flotte de test est une autre population.** Un classifieur adversarial (plis groupés par
  avion) distingue les lignes de test des lignes d'entraînement avec une AUC de **0,86** ; 63 % des
  avions de test ont des données dès la première année de l'historique, contre 0,3 % à l'entraînement.
- **La validation croisée groupée est optimiste pour cette flotte de test.** Brier du modèle brut :
  **0,140** en validation croisée groupée, **0,165** dans la même validation sur les 315 avions
  d'entraînement les plus proches du test, et **0,204** sur ces avions une fois exclus de
  l'entraînement. Environ 0,025 de l'écart tient à des avions plus difficiles ; les 0,039 restants
  viennent de l'absence d'avions semblables au test à l'entraînement (et d'un échantillon plus petit).
- **Un seul paramètre compte.** Le rétrécissement `p → 0,5 + α (p − 0,5)` ne sert à rien en
  distribution (α optimal = 1,00) mais est optimal à **α = 0,72** sur les avions proches du test.
  L'α = 0,7 soumis améliore le Brier de cet échantillon de **0,008** (IC 95 % apparié 0,002–0,014),
  à **0,197** ; avec un α choisi sur l'autre moitié des avions (validation croisée), pour ne pas le
  noter sur les lignes qui l'ont choisi, le gain est de **0,007** (0,002–0,013).
- **L'essentiel du signal transférable est l'horloge.** Sur les avions proches du test, deux
  variables temporelles seules récupèrent l'essentiel du gain du modèle complet (66 variables) face
  à une prévision constante (estimation ponctuelle **86 %** ; l'écart apparié au modèle complet
  n'est pas significatif).
- **Les leaderboards étaient trop petits pour classer.** Sur 143 lignes publiques, un score de Brier
  a une marge d'erreur de ±**0,031**. Deux scores indépendants devraient différer de plus de 0,043 ;
  une comparaison appariée sur les mêmes lignes est plus fine (environ 0,011 pour deux modèles
  proches) mais reste grossière.

## Limites

- **Le rang privé est tout aussi bruité.** La taille du leaderboard privé n'est pas connue ici. Si
  seules les lignes T et T − 24 sont notées, les 142 avions de test donnent au plus 284 lignes, donc
  au plus environ 141 pour le privé : la même marge de ±0,03. Le passage de la 22ᵉ à la 6ᵉ place est
  cohérent avec un α choisi sur une validation proche du test, mais ne prouve pas que l'approche
  était meilleure que celles des équipes classées juste autour.
- Les autres limites (une seule partition de validation, comparaisons multiples, âge approximé,
  physique sommaire) sont détaillées dans la [version anglaise](README.md#methodology-notes-and-limitations).

## Reproduire

```bash
make setup && make data && make run && make report
```

Les données du concours ne sont pas redistribuables : voir [`data/README.md`](data/README.md).
La démarche complète (en anglais) est dans [`docs/approach.md`](docs/approach.md) et dans le
[rapport interactif](https://pchambet.github.io/AIRBUS-IBM-HACKATHON-2026/).

---

Réalisé par [Pierre Chambet](https://github.com/Pchambet) — science de la décision pour des opérations sous incertitude.
