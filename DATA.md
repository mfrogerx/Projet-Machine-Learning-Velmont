# Dictionnaire des données : Valmont Express

Les données décrivent 91 jours d'activité d'une plateforme de livraison de repas dans la ville fictive de Valmont. Elles sont **simulées** : la structure est réaliste, mais aucune information n'existe ailleurs que dans ces fichiers.

Toutes les heures sont en heure locale (Valmont), sans fuseau. Les distances sont à calculer par vous : latitudes et longitudes sont en degrés décimaux (WGS84), la ville couvre environ 14 km d'est en ouest et 12 km du nord au sud.

## Fichiers

| Fichier | Lignes | Rôle |
|---------|--------|------|
| `train.csv` | environ 50 500 | Commandes du 1er septembre au 9 novembre 2025, avec la cible. |
| `test.csv` | environ 17 100 | Commandes du 10 au 30 novembre 2025, sans la cible. À prédire. |
| `sample_submission.csv` | idem test | Exemple de fichier à déposer (prédiction constante). |
| `restaurants.csv` | 120 | Table des restaurants. |
| `couriers.csv` | 420 | Table des livreurs. |
| `weather.csv` | 2 184 | Météo horaire sur les 91 jours. |
| `events.csv` | 28 | Événements urbains. |

Les fichiers `train.csv` et `test.csv` contiennent **toutes** les commandes de leur période : il n'y a pas d'échantillonnage.

## `train.csv` et `test.csv`

| Colonne | Description |
|---------|-------------|
| `order_id` | Identifiant opaque de la commande (aucune information dans sa valeur). |
| `order_time` | Moment de la commande, `AAAA-MM-JJ HH:MM:SS`. |
| `restaurant_id` | Clé vers `restaurants.csv`. |
| `customer_id` | Identifiant du client. Un client a toujours la même adresse. |
| `courier_id` | Clé vers `couriers.csv` : le livreur affecté. |
| `customer_lat`, `customer_lon` | Position du client. |
| `courier_lat`, `courier_lon` | Position du livreur au moment de l'affectation. Peut être vide (perte du signal GPS). |
| `n_items` | Nombre d'articles. |
| `order_value_eur` | Montant de la commande. |
| `has_alcohol` | 1 si la commande contient de l'alcool (contrôle d'identité à la remise). |
| `address_type` | `maison`, `immeuble_ascenseur` ou `immeuble_escalier`. |
| `floor` | Étage. Vide pour une maison, parfois vide ailleurs. |
| `special_instructions` | 1 si le client a laissé une consigne de livraison. |
| `payment_method` | Moyen de paiement. |
| `app_version` | Version de l'application cliente. |
| `legacy_eta_min` | Promesse affichée au client par le modèle actuel de la plateforme (en minutes). |

Colonnes présentes **uniquement dans `train.csv`** :

| Colonne | Description |
|---------|-------------|
| `pickup_time_min` | Minutes entre la commande et la récupération du repas par le livreur. |
| `ride_time_min` | Minutes du trajet restaurant vers client. |
| `delivered_at` | Moment de la remise. |
| `delivery_time_min` | **La cible** : minutes entre la commande et la remise au client. |

Ces colonnes ne sont connues qu'après la livraison. Elles n'existent pas dans le test : réfléchissez à ce que vous pouvez en faire.

## `restaurants.csv`

`restaurant_id`, `lat`, `lon`, `cuisine`, `price_tier` (1 à 3), `rating` (note moyenne, parfois vide), `opening_date`.

## `couriers.csv`

`courier_id`, `vehicle` (`velo`, `velo_electrique`, `scooter`, `voiture`), `hire_date`, `shift_preference` (créneau habituel : `midi`, `soir`, `midi_soir`, `weekend`, `flexible`).

## `weather.csv`

`datetime_hour` (début de l'heure), `temp_c`, `rain_mm_h`, `wind_kmh`, `humidity_pct`. Quelques heures sont vides (panne de capteur).

## `events.csv`

`event_id`, `date`, `type` (`match`, `concert`, `marathon`, `greve`, `jour_ferie`), `start_time`, `end_time`, `lat`, `lon`, `expected_attendance`. Les événements sans lieu (grève, jour férié) concernent toute la ville.

## Pièges à connaître

- Le test est **postérieur** au train. Des restaurants, des livreurs et des clients du test n'apparaissent pas dans le train.
- Il y a des valeurs manquantes, et quelques erreurs de saisie dans les positions.
- Des variables sont calculables à partir de plusieurs lignes (par exemple ce qui se passe autour d'une commande, avant elle). Les commandes du test sont toutes présentes, vous pouvez vous en servir pour décrire le contexte d'une commande, sans jamais utiliser la cible.
- La météo et les événements se joignent par date et heure.
