"""Construction des variables pour Valmont Express (train + test ensemble, sans jamais utiliser la cible du test)."""
import sys
import numpy as np
import pandas as pd

KX = 111.32 * np.cos(np.radians(46.6))  # km par degré de longitude à Valmont
KY = 110.57                             # km par degré de latitude
LON0, LAT0 = 2.30, 46.60
# Rivière : 8 ordonnées (km) aux abscisses -7..7 km, puis abscisses des 4 ponts. Estimé par river.py.
RIVER_PARAMS = np.array([-0.6255027674909379, -0.28517450518181153, 0.20395415474467624, 0.8574134212501276, 1.0173081488689732, 1.0976223605297273, 1.1916816733854192, 0.530707675561621, -1.380994589793131, -4.686534748564136, 2.2224002402526546, 5.353084839560796])
RIVER_LAT = 46.6075                     # frontière nord/sud repérée à l'exploration


def xy(lon, lat):
    return (lon - LON0) * KX, (lat - LAT0) * KY


def load(d):
    tr = pd.read_csv(d + "train.csv", parse_dates=["order_time"])
    te = pd.read_csv(d + "test.csv", parse_dates=["order_time"])
    tr["is_test"] = 0
    te["is_test"] = 1
    df = pd.concat([tr, te], ignore_index=True)
    # Erreurs de saisie : latitude et longitude inversées
    b = df.customer_lat < 40
    df.loc[b, ["customer_lat", "customer_lon"]] = df.loc[b, ["customer_lon", "customer_lat"]].values
    df["pos_fixed"] = b.astype(int)
    R = pd.read_csv(d + "restaurants.csv", parse_dates=["opening_date"])
    C = pd.read_csv(d + "couriers.csv", parse_dates=["hire_date"])
    W = pd.read_csv(d + "weather.csv", parse_dates=["datetime_hour"])
    E = pd.read_csv(d + "events.csv")
    return df, R, C, W, E


def rolling_count(times, window_min, weights=None):
    """Pour chaque instant (trié), somme des poids des lignes dans ]t-window, t[ (strictement avant)."""
    t = times.values.astype("datetime64[s]").astype(np.int64)
    w = np.ones(len(t)) if weights is None else np.asarray(weights, float)
    cs = np.concatenate([[0], np.cumsum(w)])
    lo = np.searchsorted(t, t - window_min * 60, side="left")
    hi = np.searchsorted(t, t, side="left")
    return cs[hi] - cs[lo]


def build(d):
    df, R, C, W, E = load(d)
    df = df.sort_values("order_time").reset_index(drop=True)
    df = df.merge(R.rename(columns={"lat": "rest_lat", "lon": "rest_lon"}), on="restaurant_id", how="left")
    df = df.merge(C, on="courier_id", how="left")
    t = df.order_time

    # ---------- Temps ----------
    df["hour"] = t.dt.hour + t.dt.minute / 60
    df["dow"] = t.dt.dayofweek
    df["is_weekend"] = (df.dow >= 5).astype(int)
    df["day_idx"] = (t.dt.normalize() - pd.Timestamp("2025-09-01")).dt.days

    # ---------- Géographie ----------
    rx, ry = xy(df.rest_lon, df.rest_lat)
    cx, cy = xy(df.customer_lon, df.customer_lat)
    kx_, ky_ = xy(df.courier_lon, df.courier_lat)
    df["rx"], df["ry"], df["cx"], df["cy"], df["kx"], df["ky"] = rx, ry, cx, cy, kx_, ky_
    df["d_rc"] = np.hypot(cx - rx, cy - ry)            # restaurant -> client
    df["d_rc_manh"] = (cx - rx).abs() + (cy - ry).abs()
    df["d_kr"] = np.hypot(kx_ - rx, ky_ - ry)          # livreur -> restaurant
    df["gps_missing"] = df.courier_lat.isna().astype(int)
    df["dx_rc"], df["dy_rc"] = cx - rx, cy - ry
    df["bearing_rc"] = np.degrees(np.arctan2(cy - ry, cx - rx))
    yr = (RIVER_LAT - LAT0) * KY
    df["cross_ns"] = ((ry < yr) != (cy < yr)).astype(int)
    frac = (yr - ry) / (cy - ry).replace(0, np.nan)
    df["cross_x"] = np.where(df.cross_ns == 1, rx + frac * (cx - rx), np.nan)  # où la ligne droite coupe la frontière
    df["cross_ns_kr"] = ((ry < yr) != (ky_ < yr)).astype(float)
    df.loc[df.gps_missing == 1, "cross_ns_kr"] = np.nan
    # Rivière (ligne brisée) et 4 ponts, estimés sur le train à partir des temps de trajet (voir river.py)
    p = RIVER_PARAMS
    KN, yk, bx = np.array([-7, -5, -3, -1, 1, 3, 5, 7.]), p[:8], p[8:]
    by = np.interp(bx, KN, yk)
    side = lambda x, y: (y > np.interp(x, KN, yk)).astype(float)

    def route(x1, y1, x2, y2):
        via = np.min(np.hypot(x1.values[:, None] - bx, y1.values[:, None] - by)
                     + np.hypot(x2.values[:, None] - bx, y2.values[:, None] - by), axis=1)
        straight = np.hypot(x2 - x1, y2 - y1)
        cross = side(x1, y1) != side(x2, y2)
        return np.where(cross, via, straight), cross.astype(float)

    df["route_rc"], df["river_cross"] = route(rx, ry, cx, cy)
    df["detour_rc"] = df.route_rc - df.d_rc
    df["route_kr"], df["river_cross_kr"] = route(kx_.fillna(0), ky_.fillna(0), rx, ry)
    df.loc[df.gps_missing == 1, ["route_kr", "river_cross_kr"]] = np.nan
    df["detour_kr"] = df.route_kr - df.d_kr
    df["side_r"], df["side_c"] = side(rx, ry), side(cx, cy)
    df["river_dist_c"] = cy - np.interp(cx, KN, yk)
    df["river_dist_r"] = ry - np.interp(rx, KN, yk)
    df["d_center_c"] = np.hypot(cx, cy)
    df["d_center_r"] = np.hypot(rx, ry)

    # ---------- Restaurant / livreur ----------
    df["rest_age_days"] = (t - df.opening_date).dt.days
    df["tenure_days"] = (t - df.hire_date).dt.days
    slot = np.select(
        [df.hour.between(11, 14.5), df.hour.between(18, 22.5)], ["midi", "soir"], default="autre")
    pref = df.shift_preference
    df["in_pref_shift"] = (
        ((pref == "midi") & (slot == "midi")) | ((pref == "soir") & (slot == "soir"))
        | ((pref == "midi_soir") & (slot != "autre")) | ((pref == "weekend") & (df.is_weekend == 1))
        | (pref == "flexible")).astype(int)
    # Étage manquant en immeuble : un client a toujours la même adresse, on reprend l'étage
    # de ses autres commandes (train + test). Les maisons restent sans étage.
    df["floor_missing"] = (df.floor.isna() & (df.address_type != "maison")).astype(int)
    df["floor"] = df.floor.fillna(df.groupby("customer_id").floor.transform("first"))
    df["value_per_item"] = df.order_value_eur / df.n_items
    df["floor_f"] = df.floor.fillna(0)
    df["stairs_floors"] = np.where(df.address_type == "immeuble_escalier", df.floor, 0.0)
    df["cash"] = (df.payment_method == "especes").astype(int)

    # ---------- Charge : ce qui se passe AVANT la commande (train + test) ----------
    for w in (15, 30, 60):
        df[f"city_n_{w}"] = rolling_count(t, w)
    # La fenêtre de 15 minutes est celle qui explique le mieux l'attente (testé de 3 à 120 min)
    for w in (15, 60):
        df[f"rest_n_{w}"] = 0.0
    df["rest_items_15"] = 0.0
    df["rest_gap_prev"] = np.nan
    for _, idx in df.groupby("restaurant_id").indices.items():
        tt = t.iloc[idx]
        for w in (15, 60):
            df.loc[idx, f"rest_n_{w}"] = rolling_count(tt, w)
        df.loc[idx, "rest_items_15"] = rolling_count(tt, 15, df.n_items.values[idx])
        df.loc[idx, "rest_gap_prev"] = tt.diff().dt.total_seconds().values / 60
    df["cour_n_60"] = 0.0
    df["cour_n_day"] = 0.0
    for c in ("cour_gap_prev", "cour_prev_eta", "cour_d_prev_cust", "cour_prev_d_rc"):
        df[c] = np.nan
    for _, idx in df.groupby("courier_id").indices.items():
        tt = t.iloc[idx]
        df.loc[idx, "cour_n_60"] = rolling_count(tt, 60)
        df.loc[idx, "cour_n_day"] = rolling_count(tt, 600)
        df.loc[idx, "cour_gap_prev"] = tt.diff().dt.total_seconds().values / 60
        sub = df.iloc[idx]
        df.loc[idx, "cour_prev_eta"] = sub.legacy_eta_min.shift().values
        df.loc[idx, "cour_prev_d_rc"] = sub.d_rc.shift().values
        df.loc[idx, "cour_d_prev_cust"] = np.hypot(sub.kx - sub.cx.shift(), sub.ky - sub.cy.shift()).values
    df["cour_slack"] = df.cour_gap_prev - df.cour_prev_eta   # >0 : la course précédente était censée être finie
    df["cour_exp_n"] = df.groupby("courier_id").cumcount()   # nombre de courses déjà vues
    # livreurs actifs dans l'heure -> tension offre/demande
    hkey = t.dt.floor("h")
    act = df.groupby(hkey).courier_id.nunique()
    df["couriers_hour"] = hkey.map(act).values
    df["orders_hour"] = hkey.map(df.groupby(hkey).size()).values
    df["orders_per_courier"] = df.orders_hour / df.couriers_hour
    # demande locale autour du client / restaurant (même quart d'heure)
    df["zone"] = (np.floor(rx / 2).astype(int) * 100 + np.floor(ry / 2).astype(int))
    df["zone_n_30"] = 0.0
    for _, idx in df.groupby("zone").indices.items():
        df.loc[idx, "zone_n_30"] = rolling_count(t.iloc[idx], 30)

    # ---------- Météo ----------
    W = W.sort_values("datetime_hour")
    W["weather_missing"] = W.temp_c.isna().astype(int)
    cols = ["temp_c", "rain_mm_h", "wind_kmh", "humidity_pct"]
    W[cols] = W[cols].interpolate(limit_direction="both")
    W["rain_prev_3h"] = W.rain_mm_h.rolling(3, min_periods=1).sum().shift(1)
    df = df.merge(W, left_on=hkey, right_on="datetime_hour", how="left").drop(columns=["datetime_hour"])
    df["two_wheels"] = df.vehicle.isin(["velo", "velo_electrique", "scooter"]).astype(int)
    df["rain_2w"] = df.rain_mm_h * df.two_wheels
    df["wind_bike"] = df.wind_kmh * df.vehicle.isin(["velo", "velo_electrique"]).astype(int)

    # ---------- Événements ----------
    E["start"] = pd.to_datetime(E.date + " " + E.start_time)
    E["end"] = pd.to_datetime(E.date + " " + E.end_time)
    date = df.order_time.dt.normalize()
    df["greve"] = date.isin(pd.to_datetime(E.date[E.type == "greve"])).astype(int)
    df["jour_ferie"] = date.isin(pd.to_datetime(E.date[E.type == "jour_ferie"])).astype(int)
    for c in ("ev_active", "ev_attend"):
        df[c] = 0.0
    for c in ("ev_min_from_start", "ev_min_to_end", "ev_d_rest", "ev_d_cust", "ev_d_path", "ev_d_cour"):
        df[c] = np.nan
    df["ev_type"] = "aucun"
    for _, e in E[E.lat.notna()].iterrows():
        m = (date == pd.Timestamp(e.date)).values
        if not m.any():
            continue
        ex, ey = xy(e.lon, e.lat)
        s = df.loc[m]
        tt = s.order_time
        df.loc[m, "ev_type"] = e.type
        df.loc[m, "ev_attend"] = e.expected_attendance
        df.loc[m, "ev_min_from_start"] = (tt - e.start).dt.total_seconds() / 60
        df.loc[m, "ev_min_to_end"] = (e.end - tt).dt.total_seconds() / 60
        # fenêtre large : 1h30 avant (arrivées) jusqu'à 1h30 après (sorties)
        df.loc[m, "ev_active"] = ((tt >= e.start - pd.Timedelta("90min")) & (tt <= e.end + pd.Timedelta("90min"))).astype(float)
        df.loc[m, "ev_d_rest"] = np.hypot(s.rx - ex, s.ry - ey)
        df.loc[m, "ev_d_cust"] = np.hypot(s.cx - ex, s.cy - ey)
        df.loc[m, "ev_d_cour"] = np.hypot(s.kx - ex, s.ky - ey)
        # distance de l'événement au trajet restaurant -> client (segment)
        vx, vy = s.cx - s.rx, s.cy - s.ry
        L2 = (vx ** 2 + vy ** 2).clip(lower=1e-9)
        u = (((ex - s.rx) * vx + (ey - s.ry) * vy) / L2).clip(0, 1)
        df.loc[m, "ev_d_path"] = np.hypot(s.rx + u * vx - ex, s.ry + u * vy - ey)
    df["ev_path_active"] = np.where(df.ev_active == 1, df.ev_d_path, np.nan)   # proximité, seulement si l'événement est en cours
    df["ev_pressure"] = np.where(df.ev_active == 1, df.ev_attend / (1 + df.ev_d_path) ** 2, 0.0)

    for c in ("cuisine", "vehicle", "shift_preference", "address_type", "payment_method", "ev_type"):
        df[c] = df[c].astype("category")
    return df


if __name__ == "__main__":
    out = build(sys.argv[1])
    out.to_pickle(sys.argv[2])
    print(out.shape)
