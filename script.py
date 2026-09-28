"""
Детекция ботов: сбор признаков и обучение.
- События читаются pandas, фильтруются по окну [window_start_ts, window_end_ts).
- Валидация: time-based (test позже train).
- Метрика: precision_at_recall из metric.py (официальная).
- Модели: логрег (baseline) + LightGBM (основная).
"""
import os
import random
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from metric import precision_at_recall

warnings.filterwarnings("ignore")


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


seed_everything(42)

DATA_DIR = "data"
VALID_FROM = "2026-04-17"  # граница time-based валидации


# ---------------------------------------------------------------- загрузка
def load_data():
    train = pd.read_csv(
        f"{DATA_DIR}/train.csv",
        parse_dates=["cookie_created_at", "window_start_ts", "window_end_ts"],
    )
    test = pd.read_csv(
        f"{DATA_DIR}/test.csv",
        parse_dates=["cookie_created_at", "window_start_ts", "window_end_ts"],
    )
    events = pd.read_csv(f"{DATA_DIR}/events.csv.gz", parse_dates=["event_ts"])
    return train, test, events


def cut_window(events, meta):
    """Только события внутри окна + удаление полных дублей."""
    ev = events.merge(
        meta[["cookie_id", "window_start_ts", "window_end_ts"]],
        on="cookie_id",
        how="inner",
    )
    ev = ev[(ev.event_ts >= ev.window_start_ts) & (ev.event_ts < ev.window_end_ts)]
    ev = ev.drop_duplicates()  # ← добавили
    return ev.copy()


# ---------------------------------------------------------------- фичи
def add_derived_columns(ev):
    """Нормализация platform и флаг подозрительного user_agent."""
    ev["platform_norm"] = (
        ev["platform"].astype(str).str.lower()
        .replace({"iphone": "ios", "web": "web", "desktop": "desktop",
                  "android": "android", "ios": "ios"})
    )
    # простой эвристический флаг «подозрительного» UA
    ua = ev["user_agent"].astype(str).str.lower()
    ev["ua_suspicious"] = (
        ua.str.contains("bot|spider|crawler|python|curl|wget|headless|scrapy",
                        regex=True, na=False)
    ).astype(int)
    return ev


def build_features(g):
    """Признаки по одной куке. g — события одной куки внутри окна."""
    g = g.sort_values("event_ts").copy()
    n = len(g)

    feat = {
        # --- объём и разнообразие ---
        "n_events": n,
        "n_unique_event_types": g["event_name"].nunique(),
        "n_unique_items": g["item_id"].nunique(),
        "n_unique_categories": g["item_category"].nunique(),
        "n_unique_locations": g["item_location"].nunique(),
        "n_unique_seller_types": g["seller_type"].nunique(),
        "n_unique_queries": g["search_query"].nunique(),
        "n_unique_platforms": g["platform_norm"].nunique(),
        "n_unique_user_agents": g["user_agent"].nunique(),
        # --- доли пропусков ---
        "item_id_missing_share": g["item_id"].isna().mean(),
        "category_missing_share": g["item_category"].isna().mean(),
        "location_missing_share": g["item_location"].isna().mean(),
        "seller_type_missing_share": g["seller_type"].isna().mean(),
        "query_missing_share": g["search_query"].isna().mean(),
        "search_page_missing_share": g["search_page"].isna().mean(),
        "pointer_missing_share": (
            g["pointer_x"].isna() | g["pointer_y"].isna()
        ).mean(),
        "ua_suspicious_share": g["ua_suspicious"].mean(),
    }

    # --- счётчики и доли по типам событий ---
    counts = g["event_name"].value_counts()
    for event_name, count in counts.items():
        key = str(event_name).replace(" ", "_")
        feat[f"cnt_event_{key}"] = count
        feat[f"share_event_{key}"] = count / max(n, 1)

    # --- контактные действия суммарно ---
    contact_events = {"contact_phone_show", "contact_chat_open", "contact_message_sent"}
    feat["n_contact_actions"] = int(g["event_name"].isin(contact_events).sum())

    # --- длина поискового запроса ---
    q = g["search_query"].dropna().astype(str)
    feat["mean_query_len"] = q.str.len().mean() if len(q) else 0.0
    feat["max_query_len"] = q.str.len().max() if len(q) else 0.0

    # --- eid-ratio топ-5 (n-граммы поведения) ---
    eid_counts = g["eid"].astype(str).value_counts(normalize=True)
    for i in range(1, 6):
        feat[f"eid_{i}_ratio"] = float(eid_counts.get(str(i), 0.0))

    # --- время ---
    start, end = g["event_ts"].min(), g["event_ts"].max()
    feat["activity_duration_sec"] = (end - start).total_seconds()

    intervals = g["event_ts"].diff().dt.total_seconds().dropna()
    if len(intervals):
        feat["mean_interval_sec"] = intervals.mean()
        feat["median_interval_sec"] = intervals.median()
        feat["std_interval_sec"] = intervals.std()
        feat["min_interval_sec"] = intervals.min()
        feat["max_interval_sec"] = intervals.max()
        feat["p25_interval_sec"] = np.percentile(intervals, 25)
        feat["p75_interval_sec"] = np.percentile(intervals, 75)
        feat["short_interval_share"] = (intervals < 1).mean()
        feat["long_pause_count"] = int((intervals > 300).sum())
        feat["dt_cv"] = feat["std_interval_sec"] / (feat["mean_interval_sec"] + 1e-6)
    else:
        for col in [
            "mean_interval_sec", "median_interval_sec", "std_interval_sec",
            "min_interval_sec", "max_interval_sec", "p25_interval_sec",
            "p75_interval_sec", "short_interval_share", "long_pause_count", "dt_cv",
        ]:
            feat[col] = 0.0

    feat["events_per_hour"] = n / max(feat["activity_duration_sec"] / 3600, 1 / 3600)

    # --- активность по времени суток ---
    hours = g["event_ts"].dt.hour
    feat["share_night"] = ((hours >= 0) & (hours < 6)).mean()
    feat["share_morning"] = ((hours >= 6) & (hours < 12)).mean()
    feat["share_day"] = ((hours >= 12) & (hours < 18)).mean()
    feat["share_evening"] = (hours >= 18).mean()

    # --- платформенные доли ---
    if n:
        plat_shares = g["platform_norm"].value_counts(normalize=True)
        for p in ["web", "desktop", "android", "ios"]:
            feat[f"plat_{p}"] = float(plat_shares.get(p, 0.0))

    return pd.Series(feat)


def build_all_features(ev, meta):
    ev = add_derived_columns(ev)

    rows = []
    for cid, g in ev.groupby("cookie_id", sort=False):
        s = build_features(g)
        s["cookie_id"] = cid
        rows.append(s)
    feats = pd.DataFrame(rows)

    meta_age = meta[["cookie_id", "cookie_created_at", "window_start_ts"]].copy()
    meta_age["cookie_age_sec"] = (
        meta_age["window_start_ts"] - meta_age["cookie_created_at"]
    ).dt.total_seconds()

    out = (
        meta[["cookie_id"]]
        .merge(feats, on="cookie_id", how="left")
        .merge(meta_age[["cookie_id", "cookie_age_sec"]], on="cookie_id", how="left")
    )
    return out.fillna(0)

# ---------------------------------------------------------------- обучение
def get_feature_cols(df):
    drop = {"cookie_id", "cookie_created_at", "window_start_ts",
            "window_end_ts", "target"}
    return [c for c in df.columns if c not in drop]

# ---------------------------------------------------------------- эксперименты
def train_lgbm_eval(X_tr, y_tr, X_va, y_va, seed=42):
    """Обучает LightGBM, возвращает (метрика P@R0.7, модель, предсказания)."""
    params = {
        "objective": "binary", "metric": "binary_logloss",
        "boosting_type": "gbdt", "learning_rate": 0.03,
        "num_leaves": 63, "max_depth": 8,
        "feature_fraction": 0.8, "bagging_fraction": 0.9, "bagging_freq": 5,
        "random_state": seed, "verbose": -1, "n_jobs": -1,
    }
    model = lgb.LGBMClassifier(**params, n_estimators=3000)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)],
    )
    p_va = model.predict_proba(X_va)[:, 1]
    score = precision_at_recall(y_va, p_va)
    return score, model, p_va


def select_features_by_importance(X, y, is_valid, cols, top_k=30, seed=42):
    """Обучает на всех фичах, возвращает топ-K по важности."""
    X_arr = X[cols].values
    _, model, _ = train_lgbm_eval(
        X_arr[~is_valid], y[~is_valid],
        X_arr[is_valid], y[is_valid], seed=seed,
    )
    imp = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False)
    return imp.head(top_k).index.tolist(), imp

def add_candidate_features(ev):
    """Кандидаты, отобранные по положительному delta на очищенных от дублей данных."""
    ev = ev.sort_values(["cookie_id", "event_ts"]).copy()
    ev["dt"] = ev.groupby("cookie_id")["event_ts"].diff().dt.total_seconds()

    is_contact = ev["event_name"].isin(
        {"contact_phone_show", "contact_chat_open", "contact_message_sent"}
    )
    is_phone = ev["event_name"] == "contact_phone_show"
    is_chat = ev["event_name"] == "contact_chat_open"
    ev["is_contact"] = is_contact.astype(int)
    ev["is_phone"] = is_phone.astype(int)
    ev["is_chat"] = is_chat.astype(int)

    g = ev.groupby("cookie_id")

    dt_df = pd.DataFrame({
        "p01_interval_sec": g["dt"].quantile(0.01),
        "p05_interval_sec": g["dt"].quantile(0.05),
        "share_under_0_5s": g["dt"].apply(lambda s: (s < 0.5).mean()),
    })

    contact_df = pd.DataFrame({
        "phone_show_ratio": g["is_phone"].mean(),
        "chat_open_ratio": g["is_chat"].mean(),
        "has_any_contact": g["is_contact"].max(),
    })

    div_df = pd.DataFrame({
        "unique_search_pages": g["search_page"].nunique(),
        "items_per_event": g["item_id"].nunique() / g.size().clip(lower=1),
    })

    add = pd.concat([dt_df, contact_df, div_df], axis=1).reset_index()
    return add

def test_candidates_one_by_one(Xtr, Xtr2, y, is_valid, base_cols, new_cols, score_base):
    """Добавляет кандидатов по одному, смотрит прирост P@R0.7."""
    print("\n=== Поштучный тест кандидатов ===")
    rows = [{"added": "— (base)", "n_features": len(base_cols), "P@R0.7": round(score_base, 4)}]
    best_score = score_base

    for col in new_cols:
        cols = base_cols + [col]
        score, _, _ = train_lgbm_eval(
            Xtr2[cols].values[~is_valid], y[~is_valid],
            Xtr2[cols].values[is_valid], y[is_valid],
        )
        delta = score - score_base
        rows.append({"added": col, "n_features": len(cols),
                     "P@R0.7": round(score, 4), "delta": round(delta, 4)})
        if score > best_score:
            best_score = score

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    return summary

def time_series_cv(X, y, ts, cols, n_splits=5, seed=42):
    """Time-based CV: обучение на прошлом, валидация на будущем.
    ts — массив window_start_ts.
    Фолды строятся по квантилям времени: каждый следующий блок — валидация.
    """
    print(f"\n=== TimeSeriesSplit ({n_splits} фолдов) ===")
    order = np.argsort(ts)
    X_sorted = X[cols].values[order]
    y_sorted = y[order]
    ts_sorted = ts[order]

    # делим на n_splits+1 блоков по времени; i-й фолд: train = [0..i], val = (i..i+1]
    n = len(y_sorted)
    fold_size = n // (n_splits + 1)
    scores = []

    for i in range(1, n_splits + 1):
        tr_end = i * fold_size
        va_end = (i + 1) * fold_size if i < n_splits else n

        X_tr, y_tr = X_sorted[:tr_end], y_sorted[:tr_end]
        X_va, y_va = X_sorted[tr_end:va_end], y_sorted[tr_end:va_end]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
            print(f"  Фолд {i}: пропущен (один класс в train/val)")
            continue

        score, _, _ = train_lgbm_eval(X_tr, y_tr, X_va, y_va, seed=seed)
        scores.append(score)
        print(f"  Фолд {i}: train={len(y_tr)}, val={len(y_va)}, "
              f"val_from={pd.Timestamp(ts_sorted[tr_end]).date()}, "
              f"P@R0.7 = {score:.4f}")

    if scores:
        print(f"\n  Среднее P@R0.7: {np.mean(scores):.4f}")
        print(f"  Std:           {np.std(scores):.4f}")
        print(f"  Min / Max:     {np.min(scores):.4f} / {np.max(scores):.4f}")
    return scores

def main():
    print("Загрузка...")
    train, test, events = load_data()
    print("shapes:", train.shape, test.shape, events.shape)
    print("доля ботов в train:", round(train.target.mean(), 4))

    print("Фильтр по окну...")
    ev_tr = cut_window(events, train)
    ev_te = cut_window(events, test)
    print("событий в окнах:", len(ev_tr), len(ev_te))

    print("Сбор признаков...")
    Xtr = build_all_features(ev_tr, train)
    Xte = build_all_features(ev_te, test)
    print("Xtr (v1):", Xtr.shape, "Xte (v1):", Xte.shape)

    # --- таргет и валидация по времени ---
    y = train.set_index("cookie_id").loc[Xtr["cookie_id"], "target"].values
    is_valid = (
        train.set_index("cookie_id").loc[Xtr["cookie_id"], "window_start_ts"]
        >= pd.Timestamp(VALID_FROM)
    ).values

    base_cols = [c for c in Xtr.columns if c != "cookie_id"]

    # --- baseline: логрег на всех фичах ---
    print("\n--- baseline: LogisticRegression ---")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=42)),
    ])
    pipe.fit(Xtr[base_cols].values[~is_valid], y[~is_valid])
    p_lr = pipe.predict_proba(Xtr[base_cols].values[is_valid])[:, 1]
    score_lr = precision_at_recall(y[is_valid], p_lr)
    print(f"P@R0.7 (logreg, все {len(base_cols)} фич): {score_lr:.4f}")

    # --- эксперимент 1: LightGBM на всех фичах ---
    print("\n=== Эксперимент 1: LightGBM, все фичи ===")
    score_all, model_all, _ = train_lgbm_eval(
        Xtr[base_cols].values[~is_valid], y[~is_valid],
        Xtr[base_cols].values[is_valid], y[is_valid],
    )
    print(f"P@R0.7: {score_all:.4f}")

    # --- эксперимент 2: топ-30 по важности ---
    print("\n=== Эксперимент 2: топ-30 по важности ===")
    top_cols, imp = select_features_by_importance(
        Xtr, y, is_valid, base_cols, top_k=30,
    )
    score_top, _, _ = train_lgbm_eval(
        Xtr[top_cols].values[~is_valid], y[~is_valid],
        Xtr[top_cols].values[is_valid], y[is_valid],
    )
    print(f"P@R0.7: {score_top:.4f}")

    # --- эксперимент 3: + новые фичи-кандидаты ---
    print("\n=== Эксперимент 3: все фичи + кандидаты ===")
    add_tr = add_candidate_features(ev_tr)
    add_te = add_candidate_features(ev_te)
    Xtr2 = Xtr.merge(add_tr, on="cookie_id", how="left").fillna(0)
    Xte2 = Xte.merge(add_te, on="cookie_id", how="left").fillna(0)
    new_cols = [c for c in add_tr.columns if c != "cookie_id"]
    
    # проверка корреляции интервальных фич
    check_cols = [c for c in ["p10_interval_sec", "min_interval_sec",
                              "median_interval_sec", "mean_interval_sec",
                              "p25_interval_sec", "p75_interval_sec"]
                  if c in Xtr2.columns]
    if len(check_cols) >= 2:
        print("\nКорреляция интервальных фич:")
        print(Xtr2[check_cols].corr().round(3))
    
    cols_v2 = base_cols + new_cols

    score_v2, _, _ = train_lgbm_eval(
        Xtr2[cols_v2].values[~is_valid], y[~is_valid],
        Xtr2[cols_v2].values[is_valid], y[is_valid],
    )
    print(f"P@R0.7: {score_v2:.4f}")

    # --- поштучный тест кандидатов ---
    cand_summary = test_candidates_one_by_one(
        Xtr, Xtr2, y, is_valid, base_cols, new_cols, score_all,
    )
    # --- TimeSeriesSplit на финальном наборе ---
    ts_tr = train.set_index("cookie_id").loc[Xtr2["cookie_id"], "window_start_ts"].values
    cv_scores = time_series_cv(Xtr2, y, ts_tr, cols_v2, n_splits=5)

    # --- сводка ---
    print("\n=== Сводка экспериментов ===")
    summary = pd.DataFrame([
        {"эксперимент": "logreg (baseline)", "n_features": len(base_cols), "P@R0.7": round(score_lr, 4)},
        {"эксперимент": "lightgbm, все фичи (v1)", "n_features": len(base_cols), "P@R0.7": round(score_all, 4)},
        {"эксперимент": "lightgbm, top-30", "n_features": 30, "P@R0.7": round(score_top, 4)},
        {"эксперимент": "lightgbm, все + новые (v2)", "n_features": len(cols_v2), "P@R0.7": round(score_v2, 4)},
    ])
    print(summary.to_string(index=False))
    print("\nТоп-20 фич:")
    print(imp.head(20))

    # --- выбор лучшего варианта (по метрике, при равенстве — меньше фич) ---
    candidates = [
        ("v1_all", base_cols, Xtr, Xte, score_all),
        ("v1_top30", top_cols, Xtr, Xte, score_top),
        ("v2_all_plus_new", cols_v2, Xtr2, Xte2, score_v2),
    ]
    best_name, best_cols, best_Xtr, best_Xte, best_score = max(
        candidates, key=lambda t: (t[4], -len(t[1]))
    )
    print(f"\nЛучший вариант: {best_name}, P@R0.7 = {best_score:.4f}, "
          f"признаков: {len(best_cols)}")

    # --- финальная модель на всём train ---
    print("\nОбучение финальной модели на всём train...")
    final = lgb.LGBMClassifier(
        objective="binary", metric="binary_logloss", boosting_type="gbdt",
        learning_rate=0.03, num_leaves=63, max_depth=8,
        feature_fraction=0.8, bagging_fraction=0.9, bagging_freq=5,
        random_state=42, verbose=-1, n_jobs=-1, n_estimators=1000,
    )
    final.fit(best_Xtr[best_cols].values, y)
    test_pred = final.predict_proba(best_Xte[best_cols].values)[:, 1]

    sub = pd.DataFrame({"cookie_id": best_Xte["cookie_id"], "score": test_pred})
    assert len(sub) == len(test), "длина сабмита не совпадает с test"
    assert sub["score"].between(0, 1).all(), "score вне [0,1]"
    sub.to_csv("submission.csv", index=False)
    print("Готово: submission.csv")

if __name__ == "__main__":
    main()