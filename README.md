# Valmont Express : prédiction du temps de livraison (soumission v2)

Score public : 4,999. Validation (3 dernières semaines du train) : 4,52. Second découpage : 4,58.

## Lancer

```bash
pip install -r requirements.txt
bash run_all.sh "chemin/vers/le/dossier/des/csv/"
```

Le dossier doit contenir `train.csv`, `test.csv`, `restaurants.csv`, `couriers.csv`, `weather.csv`, `events.csv`
(mettre le `/` final). Le résultat est `submission_v2.csv`.

## Fichiers

| Fichier | Rôle |
|---|---|
| `features.py` | Nettoyage et construction des variables, train et test ensemble, sans la cible du test |
| `common.py` | Liste des variables, écarts historiques par restaurant / livreur / client, découpages temporels |
| `prep.py` | Prépare les trois jeux : B (validation), A (contrôle), final (tout le train -> test) |
| `run_model.py` | Entraîne un modèle (`lgb`, `xgb`, `cat`, et pour comparaison `ridge`, `rf`, `mlp`) sur un découpage |
| `final_blend.py` | Affiche les RMSE de validation et écrit la soumission (moyenne des 3 boostings) |
| `river.py` | Trace de la façon dont la rivière et les 4 ponts ont été estimés. Son résultat est déjà recopié dans `features.py` (`RIVER_PARAMS`) ; inutile de le relancer |

## Le modèle

- Trois modèles de gradient boosting : LightGBM, XGBoost, CatBoost, en moyenne simple.
- Chacun prédit trois étapes puis les additionne : attente au restaurant, trajet, remise au client.
  Les colonnes `pickup_time_min` et `ride_time_min` servent donc de cibles intermédiaires, jamais d'entrées.
- Attente et trajet sont appris en deux versions (minutes et logarithme), moyennées.
- Remise : perte de Huber (insensible aux remises anormalement longues), puis ajout de l'écart moyen.

## Validation

Le test est dans le futur du train : on apprend sur le passé et on valide sur les semaines suivantes.
- B : apprentissage avant le 20/10, validation du 20/10 au 09/11. Sert aussi à fixer le nombre d'arbres.
- A : apprentissage avant le 06/10, validation du 06/10 au 19/10. Sert de contrôle indépendant.

## Données sales ou manquantes

- Latitude / longitude inversées (285 commandes) : remises dans l'ordre.
- Étage manquant en immeuble : repris des autres commandes du même client (adresse fixe).
- GPS du livreur manquant (4 %) : laissé vide + indicateur.
- Météo manquante (24 heures) : interpolation + indicateur.
- Restaurants, livreurs, clients jamais vus : écart nul, donc moyenne de leur catégorie. À l'entraînement,
  l'historique des entités récentes est masqué pour que le modèle apprenne ce cas.
- Variables écartées car elles ne font que suivre le temps : `app_version`, température, humidité.

## Ce qui n'a pas marché

- v3 : extrapoler l'effet de la pluie au-delà de ce que contient le train (score public 5,024).
- Forêt aléatoire, Ridge, réseau de neurones : moins bons seuls et dans la moyenne.
- Modèle dédié aux nouveaux restaurants, loi fixe pour les grèves : pas de gain net en validation.
