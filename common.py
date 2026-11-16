"""Éléments communs à tous les modèles : liste des variables, encodages historiques, découpages temporels."""
import numpy as np
import pandas as pd

BASE = [
    "hour", "dow", "is_weekend", "rx", "ry", "cx", "cy", "kx", "ky", "d_rc", "d_rc_manh", "d_kr", "gps_missing",
    "dx_rc", "dy_rc", "bearing_rc", "route_rc", "river_cross", "detour_rc", "route_kr", "river_cross_kr", "detour_kr",
    "side_r", "side_c", "river_dist_c", "river_dist_r", "d_center_c", "d_center_r",
    "cuisine", "price_tier", "rating", "rest_age_days", "vehicle", "tenure_days", "shift_preference", "in_pref_shift",
    "n_items", "order_value_eur", "value_per_item", "has_alcohol", "address_type", "floor", "floor_missing",
    "stairs_floors", "special_instructions", "payment_method", "cash", "legacy_eta_min",
    "city_n_15", "city_n_30", "city_n_60", "rest_n_15", "rest_n_60", "rest_items_15",
    "rest_gap_prev", "cour_n_60", "cour_n_day", "cour_gap_prev", "cour_slack", "cour_d_prev_cust", "cour_prev_d_rc",
    "couriers_hour", "orders_hour", "orders_per_courier", "zone_n_30",
    "rain_mm_h", "wind_kmh", "rain_prev_3h", "rain_2w", "wind_bike", "weather_missing",
    "greve", "jour_ferie", "ev_active", "ev_attend", "ev_type", "ev_min_from_start", "ev_min_to_end",
    "ev_d_rest", "ev_d_cust", "ev_d_path", "ev_d_cour", "ev_path_active", "ev_pressure",
]
TE = ["te_rest_pick", "te_rest_dev", "te_rest_n", "te_cour_pace", "te_cour_dev", "te_cour_pick", "te_cour_n",
      "te_cust_hand", "te_cust_n", "eta_rc", "eta_kr"]
FEATS = BASE + TE
CATS = ["cuisine", "vehicle", "shift_preference", "address_type", "payment_method", "ev_type"]
TARGETS = ["delivery_time_min", "pickup_time_min", "ride_time_min", "hand"]

# Découpages temporels : on apprend sur le passé, on valide sur les semaines suivantes (comme le test).
FOLDS = {
    "B": ("2025-10-20", "2025-11-10"),   # validation principale : 3 dernières semaines du train
    "A": ("2025-10-06", "2025-10-20"),   # second découpage : sert à vérifier que les combinaisons tiennent
}


N_CAP = {"restaurant_id": 150, "courier_id": 60, "customer_id": 15}
JUNIOR_DAYS = {"courier_id": 180}   # un livreur de moins de 6 mois roule ~12 % moins vite


def add_te(df, fit_mask, drop=0.08, young_drop=0.8):
    """Écarts historiques par restaurant / livreur / client, calculés uniquement sur les lignes fit_mask.
    - L'écart est mesuré par rapport à la moyenne du groupe (cuisine, véhicule, type d'adresse) : une entité
      jamais vue a un écart de 0 et retombe sur la moyenne de sa catégorie.
    - Pour les lignes de fit : calcul « hors de sa semaine » (pas de fuite de la cible).
    - drop : part des (entité, semaine) du fit traitées comme inconnues, pour que le modèle apprenne
      à gérer les nouveaux restaurants / livreurs / clients du test."""
    df = df.copy()
    df["hand"] = df.delivery_time_min - df.pickup_time_min - df.ride_time_min
    df["pace"] = (df.ride_time_min / df.route_rc.clip(lower=0.3)).clip(upper=15)
    df["all"] = 0
    specs = [("restaurant_id", "cuisine", "pickup_time_min", "te_rest_dev", "te_rest_n", 10, "te_rest_pick"),
             ("courier_id", "vehicle", "pace", "te_cour_dev", "te_cour_n", 10, "te_cour_pace"),
             ("courier_id", "all", "pickup_time_min", "te_cour_pick", None, 30, None),
             ("customer_id", "address_type", "hand", "te_cust_hand", "te_cust_n", 5, None)]
    fit = df[fit_mask]
    block = (df.day_idx // 7)
    young_cols = {"restaurant_id": "rest_age_days", "courier_id": "tenure_days", "customer_id": None}
    for key, grp, val, name, nname, prior, absname in specs:
        young_col = young_cols[key]
        v = fit[val].clip(upper=fit[val].quantile(0.995))      # les valeurs extrêmes ne tirent pas les moyennes
        gmean = v.groupby(fit[grp], observed=True).mean()
        base = df[grp].map(gmean).astype(float)
        if key in JUNIOR_DAYS:   # correction « débutant » : les livreurs récents ont leur propre moyenne
            junior = (df[young_col] < JUNIOR_DAYS[key]).astype(int)
            jmean = (v - base[fit_mask]).groupby(junior[fit_mask]).mean()
            base = base + junior.map(jmean).fillna(0.0)
        r = v - base[fit_mask]
        s, n = r.groupby(fit[key]).sum(), r.groupby(fit[key]).count()
        S, N = df[key].map(s).fillna(0).values, df[key].map(n).fillna(0).values
        kb = pd.MultiIndex.from_arrays([fit[key], block[fit_mask]])
        sb, nb = r.groupby(kb).sum(), r.groupby(kb).count()
        allk = pd.MultiIndex.from_arrays([df[key], block])
        S2 = S - np.where(fit_mask, sb.reindex(allk).fillna(0).values, 0)
        N2 = N - np.where(fit_mask, nb.reindex(allk).fillna(0).values, 0)
        if drop > 0:
            hcode = pd.util.hash_pandas_object(
                pd.DataFrame({"k": df[key].astype(str), "b": block.values}), index=False).values
            # Dans le test, une entité inconnue est toujours une entité récente (restaurant qui vient
            # d'ouvrir, livreur qui vient d'être recruté) : on masque donc 80 % des semaines des
            # entités de moins de 3 semaines, en plus du masquage aléatoire.
            young = (df[young_col] < 21).values if young_col else np.zeros(len(df), bool)
            hide = fit_mask & ((hcode % 1000) < np.where(young, young_drop, drop) * 1000)
            S2, N2 = np.where(hide, 0, S2), np.where(hide, 0, N2)
        df[name] = S2 / (N2 + prior)
        if nname:
            # plafonné : dans le test les compteurs dépassent ce qui a été vu à l'apprentissage,
            # et un modèle à arbres se comporte mal sur des valeurs hors plage
            df[nname] = np.minimum(N2, N_CAP[key])
        if absname:
            df[absname] = base.values + df[name]
    # temps de trajet attendus = distance par les ponts x allure habituelle du livreur
    df["eta_rc"] = df.route_rc * df.te_cour_pace
    df["eta_kr"] = df.route_kr * df.te_cour_pace
    return df


def make_split(df, name):
    """Renvoie (X_fit, Y_fit, X_pred, Y_pred, ids_pred) pour un découpage 'A', 'B' ou 'final'."""
    tr = (df.is_test == 0).values
    if name == "final":
        fit, pred = tr, ~tr
    else:
        lo, hi = map(pd.Timestamp, FOLDS[name])
        fit = tr & (df.order_time < lo).values
        pred = tr & (df.order_time >= lo).values & (df.order_time < hi).values
    d = add_te(df, fit)
    return (d.loc[fit, FEATS], d.loc[fit, TARGETS], d.loc[pred, FEATS],
            d.loc[pred, TARGETS] if name != "final" else None, d.loc[pred, "order_id"].values)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
