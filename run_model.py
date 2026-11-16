"""Entraîne un modèle sur un découpage et sauvegarde ses prédictions.
Usage : python run_model.py {ridge|rf|mlp|lgb|xgb|cat} {B|A|final}
Chaque modèle prédit : la cible directement, puis ses 3 composantes (attente resto, trajet, remise).
Les modèles de boosting ont en plus une variante entraînée sur le logarithme des temps."""
import sys, os, json, pickle, time, warnings
import numpy as np
import pandas as pd
from common import CATS, TARGETS, rmse

warnings.filterwarnings("ignore")
MODEL, SPLIT = sys.argv[1], sys.argv[2]
SEEDS = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0]
Xf, Yf, Xp, Yp, ids = pickle.load(open(f"split_{SPLIT}.pkl", "rb"))
os.makedirs("preds", exist_ok=True)
ONLY = os.environ.get("ONLY", ",".join(TARGETS)).split(",")   # pour ne relancer qu'une partie des cibles
ITERS_FILE = f"iters_{MODEL}.json"
ES = SPLIT == "B"                                   # arrêt anticipé uniquement sur la validation principale
iters = json.load(open(ITERS_FILE)) if os.path.exists(ITERS_FILE) else {}
SCALE = {"B": 1.0, "A": 0.85, "final": 1.2}[SPLIT]  # nombre d'arbres ajusté à la taille de l'apprentissage
SHIFT = {"delivery_time_min": 0.0, "pickup_time_min": 0.0, "ride_time_min": 1.0}
SHORT = {"delivery_time_min": "direct", "pickup_time_min": "pickup", "ride_time_min": "ride", "hand": "hand"}


# ------------------------------------------------------------------ boosting
def fit_gbm(tgt, y, yv, seed, huber):
    key = tgt
    n_fixed = None if ES else max(50, int(iters[key] * SCALE))
    if MODEL == "lgb":
        import lightgbm as lgb
        prm = dict(objective="huber" if huber else "regression", alpha=3.0, learning_rate=0.03, num_leaves=15,
                   min_child_samples=40, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                   verbose=-1, seed=seed, num_threads=2)
        ds = lgb.Dataset(Xf, y)
        if ES:
            m = lgb.train(prm, ds, 4000, valid_sets=[lgb.Dataset(Xp, yv)],
                          callbacks=[lgb.early_stopping(150, verbose=False)])
            return m.predict(Xp, num_iteration=m.best_iteration), m.predict(Xf, num_iteration=m.best_iteration), m.best_iteration
        m = lgb.train(prm, ds, n_fixed)
        return m.predict(Xp), m.predict(Xf), n_fixed
    if MODEL == "xgb":
        import xgboost as xgb
        prm = dict(objective="reg:pseudohubererror" if huber else "reg:squarederror", huber_slope=3.0, eta=0.03,
                   max_depth=5, min_child_weight=20, subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0,
                   tree_method="hist", max_cat_to_onehot=16, nthread=2, seed=seed,
                   base_score=float(np.median(y)))
        df_, dp_ = xgb.DMatrix(Xf, y, enable_categorical=True), xgb.DMatrix(Xp, yv, enable_categorical=True)
        if ES:
            m = xgb.train(prm, df_, 5000, evals=[(dp_, "v")], early_stopping_rounds=150, verbose_eval=False)
            it = (0, m.best_iteration + 1)
            return m.predict(dp_, iteration_range=it), m.predict(df_, iteration_range=it), m.best_iteration + 1
        m = xgb.train(prm, df_, n_fixed)
        return m.predict(dp_), m.predict(df_), n_fixed
    if MODEL == "cat":
        from catboost import CatBoostRegressor, Pool
        cf = lambda X: X.assign(**{c: X[c].astype(str) for c in CATS})
        pf, pp = Pool(cf(Xf), y, cat_features=CATS), Pool(cf(Xp), yv, cat_features=CATS)
        m = CatBoostRegressor(loss_function="Huber:delta=3.0" if huber else "RMSE", learning_rate=0.06, depth=6,
                              l2_leaf_reg=3.0, iterations=5000 if ES else n_fixed, random_seed=seed, thread_count=2,
                              verbose=False, allow_writing_files=False)
        if ES:
            m.fit(pf, eval_set=pp, early_stopping_rounds=150, use_best_model=True)
            return m.predict(pp), m.predict(pf), m.get_best_iteration() + 1
        m.fit(pf)
        return m.predict(pp), m.predict(pf), n_fixed


def run_gbm():
    out = {}
    for tgt in ONLY:
        variants = [("", False)] + ([("_log", True)] if tgt != "hand" else [])
        for suffix, logt in variants:
            key = tgt + suffix
            p = np.zeros(len(Xp))
            for sd in SEEDS:
                y = Yf[tgt].values
                yv = Yp[tgt].values if Yp is not None else None
                if logt:
                    sh = SHIFT[tgt]
                    pp, pf, it = fit_gbm(key, np.log(y + sh), np.log(yv + sh) if yv is not None else None, sd, False)
                    smear = np.mean(np.exp(np.log(y + sh) - pf))      # correction du biais de retour à l'échelle
                    pp = np.exp(pp) * smear - sh
                else:
                    pp, pf, it = fit_gbm(key, y, yv, sd, tgt == "hand")
                    if tgt == "hand":
                        # La perte de Huber ignore les remises anormalement longues : elle vise le cas
                        # « normal ». La RMSE demande la moyenne, on rajoute donc l'écart moyen observé.
                        pp = pp + np.mean(y - pf)
                p += pp / len(SEEDS)
            if ES:
                iters[key] = int(it)
            out[SHORT[tgt] + suffix] = p
            print(f"  {key:24s} arbres={it:5d}", f"rmse={rmse(Yp[tgt], p):.4f}" if Yp is not None else "", flush=True)
    return out


# ------------------------------------------------------------------ modèles scikit-learn
def design():
    """Les modèles scikit-learn n'acceptent ni valeurs manquantes ni catégories :
    indicatrices pour les catégories, médiane + colonne « manquant » pour les trous."""
    X = pd.concat([Xf, Xp])
    num = [c for c in X.columns if c not in CATS]
    D = pd.get_dummies(X[CATS], dtype=float)
    N = X[num].astype(float)
    miss = [c for c in num if N[c].isna().any()]
    M = N[miss].isna().astype(float).add_suffix("_na")
    N = N.fillna(N.iloc[:len(Xf)].median())
    Z = pd.concat([N, M, D], axis=1)
    Z = Z.loc[:, Z.iloc[:len(Xf)].std() > 0]
    return Z.iloc[:len(Xf)].values, Z.iloc[len(Xf):].values


def run_sklearn():
    from sklearn.preprocessing import StandardScaler
    A, B = design()
    out = {}
    if MODEL in ("ridge", "mlp"):
        sc = StandardScaler().fit(A)
        A, B = np.clip(sc.transform(A), -6, 6), np.clip(sc.transform(B), -6, 6)
    for tgt in TARGETS:
        y = Yf[tgt].values
        p = np.zeros(len(B))
        for sd in SEEDS:
            if MODEL == "ridge":
                from sklearn.linear_model import RidgeCV
                m = RidgeCV(alphas=np.logspace(-1, 3, 9)).fit(A, y)
                pp = m.predict(B)
            elif MODEL == "rf":
                from sklearn.ensemble import RandomForestRegressor
                m = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.33, max_samples=0.6,
                                          n_jobs=2, random_state=sd).fit(A, y)
                pp = m.predict(B)
            elif MODEL == "mlp":
                from sklearn.neural_network import MLPRegressor
                yc = np.minimum(y, np.quantile(y, 0.999))           # un réseau supporte mal les valeurs extrêmes
                mu, sg = yc.mean(), yc.std()
                m = MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-2, learning_rate_init=1e-3, batch_size=256,
                                 max_iter=80, early_stopping=True, validation_fraction=0.1, n_iter_no_change=8,
                                 random_state=sd).fit(A, (yc - mu) / sg)
                pp = m.predict(B) * sg + mu
            p += pp / len(SEEDS)
        out[SHORT[tgt]] = p
        print(f"  {tgt:24s}", f"rmse={rmse(Yp[tgt], p):.4f}" if Yp is not None else "", flush=True)
    return out


t0 = time.time()
print(f"== {MODEL} / découpage {SPLIT}")
out = run_gbm() if MODEL in ("lgb", "xgb", "cat") else run_sklearn()
if len(ONLY) < len(TARGETS):
    prev = pickle.load(open(f"preds/{SPLIT}_{MODEL}.pkl", "rb"))
    prev.update(out)
    out = prev
out["comp"] = out["pickup"] + out["ride"] + out["hand"]
if "pickup_log" in out:
    out["comp_log"] = out["pickup_log"] + out["ride_log"] + out["hand"]
if Yp is not None:
    y = Yp["delivery_time_min"]
    print("  >> direct", round(rmse(y, out["direct"]), 4), "| composantes", round(rmse(y, out["comp"]), 4),
          ("| composantes log " + str(round(rmse(y, out["comp_log"]), 4))) if "comp_log" in out else "")
if ES and MODEL in ("lgb", "xgb", "cat"):
    json.dump(iters, open(ITERS_FILE, "w"))
pickle.dump(out, open(f"preds/{SPLIT}_{MODEL}.pkl", "wb"))
print(f"  temps : {time.time() - t0:.0f}s")
