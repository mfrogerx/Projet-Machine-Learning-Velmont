"""Soumission finale : moyenne simple LightGBM + XGBoost + CatBoost (composantes, échelle normale et log)."""
import pickle, sys
import numpy as np, pandas as pd
from common import rmse
G, K = ["lgb", "xgb", "cat"], ["comp", "comp_log"]
for S in ("B", "A"):
    Y = pickle.load(open(f"split_{S}.pkl", "rb"))[3]
    y = Y.delivery_time_min.values
    import os
    P = {m: pickle.load(open(f"preds/{S}_{m}.pkl", "rb")) for m in G + ["ridge", "rf", "mlp"] if os.path.exists(f"preds/{S}_{m}.pkl")}
    print(f"--- découpage {S} ({len(y)} commandes)")
    for m in [m for m in ("ridge", "mlp", "rf") if m in P]:
        print(f"  {m:28s} {min(rmse(y, P[m]['direct']), rmse(y, P[m]['comp'])):.3f}")
    for m in G:
        print(f"  {m:28s} {rmse(y, np.mean([P[m][k] for k in K], axis=0)):.3f}")
    for a, b in (("lgb", "xgb"), ("lgb", "cat"), ("xgb", "cat")):
        print(f"  {a + ' + ' + b:28s} {rmse(y, np.mean([P[m][k] for m in (a, b) for k in K], axis=0)):.3f}")
    blend = np.mean([P[m][k] for m in G for k in K], axis=0)
    e = y - blend
    print(f"  {'lgb + xgb + cat (RETENU)':28s} {rmse(y, blend):.3f}   MAE {np.abs(e).mean():.2f}  biais {e.mean():+.2f}  à ±5 min : {(np.abs(e) < 5).mean():.1%}")
ids = pickle.load(open("split_final.pkl", "rb"))[4]
P = {m: pickle.load(open(f"preds/final_{m}.pkl", "rb")) for m in G}
pred = np.mean([P[m][k] for m in G for k in K], axis=0)
print("corrélations entre modèles (test) :", np.round(np.corrcoef([np.mean([P[m][k] for k in K], axis=0) for m in G]), 4).tolist())
sub = pd.DataFrame({"order_id": ids, "delivery_time_min": np.clip(pred, 5, 200).round(2)})
sub.to_csv(sys.argv[1], index=False)
print("écrit :", sub.shape, sub.delivery_time_min.describe().round(2).to_dict())
