#!/bin/bash
# Reproduit submission_v2.csv de bout en bout. Usage : bash run_all.sh chemin/vers/les/csv/
# Compter 30 à 60 minutes selon la machine (CatBoost est le plus long).
set -e
DATA=${1:-./}
python features.py "$DATA" feats.pkl          # 1. variables (train + test)
python prep.py feats.pkl                      # 2. découpages : B (validation), A (contrôle), final
for m in lgb xgb cat; do
  python run_model.py $m B                    # 3. validation principale : fixe le nombre d'arbres
  python run_model.py $m A                    # 4. second découpage : contrôle
done
python run_model.py lgb final 0,1,2           # 5. entraînement sur tout le train, prédiction du test
python run_model.py xgb final 0,1,2
python run_model.py cat final 0,1
python final_blend.py submission_v2.csv       # 6. moyenne des 3 modèles + fichier à déposer
