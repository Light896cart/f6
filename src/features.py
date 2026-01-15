"""
Назначение:
Основной модуль препроцессинга признаков: обработка build-информации, флагов, пакетов, создание комбинированных признаков (например, emulator_pattern, risky_combo_1).
Статус: Актуальный - может использоваться.
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
import math
from typing import Optional

# === Константы ===
SUSPICIOUS_HOST_PATTERNS = ["mi-server", "ota-bd", "pangu", "buildsrv", "cn-", "dg02", "pool", "kvm"]
COMMON_SERIAL = "UNi0qUHCa4lILJSrMktaJ0+c7WY="

LEGIT_PACKAGES_PREFIXES = {
    "com.whatsapp",
    "org.telegram.messenger",
    "com.viber.voip",
    "ru.sberbankmobile",
    "com.tinkoff.android",
    "com.google.android.gm",
    "com.facebook.katana",
    "com.instagram.android",
    "com.google.android.youtube",
    "com.android.chrome",
    "com.yandex.browser",
}

SUSPICIOUS_PACKAGES_SUBSTR = {
    "bluestacks",
    "genymotion",
    "appium",
    "androidemu",
    "microvirt",
    "experitest",
    "andyroid",
    "ldmnq",
    "nox",
    "memu",
}


def safe_bool_to_int(val) -> int:
    if pd.isna(val):
        return 0
    if isinstance(val, str):
        return 1 if val.lower() in ("true", "1") else 0
    return int(bool(val))


def parse_debugger(x) -> int:
    if pd.isna(x):
        return 0
    s = str(x).lower()
    return 1 if "true" in s else 0


def has_test_keys(val) -> bool:
    return pd.notna(val) and "test-keys" in str(val).lower()


def extract_key_type(val) -> str:
    if pd.isna(val):
        return "missing"
    s = str(val).lower()
    if "test-keys" in s:
        return "test"
    elif "release-keys" in s:
        return "release"
    else:
        return "unknown"


def entropy(s):
    if not s or pd.isna(s):
        return 0.0
    p, lns = Counter(s), float(len(s))
    return -sum(count / lns * math.log(count / lns + 1e-9) for count in p.values())


def preprocess_features(
        df: pd.DataFrame,
        packages_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Полноценный препроцессинг датасета с поддержкой пакетов и продвинутых признаков.

    Вход:
        df — DataFrame с device_info (должен содержать 'agent_id', 'target', и build-колонки)
        packages_path — путь к android_packages.csv.gz (опционально, но нужен для frac_legit_certs и других признаков)

    Выход:
        Обработанный DataFrame без чувствительных/идентифицирующих полей, готовый к обучению.
    """
    df = df.copy()

    # Целевая переменная
    if "target" in df.columns:
        df["target"] = df["target"].map({"F": 1, "G": 0})

    # === Обработка пакетов (если указан путь) ===
    if packages_path is not None:
        # Загружаем пакеты
        pkg_full = pd.read_csv(packages_path, compression="gzip", usecols=["agent_id", "package", "data"])
        pkg_full["cert"] = pkg_full["data"].str.extract(r'"cert"\s*:\s*"([^"]*)"')
        pkg_full["installed"] = pkg_full["data"].str.extract(r'"installed"\s*:\s*"([^"]*)"')

        # --- Базовые агрегаты ---
        n_packages = pkg_full.groupby("agent_id").size()
        n_unique_certs = pkg_full.groupby("agent_id")["cert"].nunique()

        is_legit = pd.Series(False, index=pkg_full.index)
        for prefix in LEGIT_PACKAGES_PREFIXES:
            is_legit |= pkg_full["package"].str.startswith(prefix, na=False)
        n_legit_apps = pkg_full[is_legit].groupby("agent_id").size().reindex(n_packages.index, fill_value=0)

        is_susp = pd.Series(False, index=pkg_full.index)
        for substr in SUSPICIOUS_PACKAGES_SUBSTR:
            is_susp |= pkg_full["package"].str.contains(substr, case=False, na=False)
        has_suspicious_pkg = (
            pkg_full[is_susp].groupby("agent_id").size()
            .fillna(0).clip(upper=1).astype(int)
            .reindex(n_packages.index, fill_value=0)
        )

        # --- Временные признаки ---
        pkg_full["installed_dt"] = pd.to_datetime(pkg_full["installed"], errors="coerce", unit="ms")
        min_t = pkg_full.groupby("agent_id")["installed_dt"].min()
        max_t = pkg_full.groupby("agent_id")["installed_dt"].max()
        span = (max_t - min_t).dt.total_seconds().fillna(0) / 3600.0
        install_span = span.reindex(n_packages.index, fill_value=0.0)

        # --- Сбор базовых признаков ---
        pkg_feats = pd.DataFrame({
            "agent_id": n_packages.index,
            "n_packages": n_packages.values,
            "n_unique_certs": n_unique_certs.reindex(n_packages.index, fill_value=0).values,
            "n_legit_apps": n_legit_apps.values,
            "has_suspicious_pkg": has_suspicious_pkg.values,
            "install_span_hours": install_span.values,
        })

        # Мержим в основной df
        df = df.merge(pkg_feats, on="agent_id", how="left")

        # --- Расчёт frac_legit_certs (только если есть target) ---
        if "target" in df.columns:
            # Добавляем target к пакетам
            pkg_with_target = pkg_full.merge(df[["agent_id", "target"]], on="agent_id", how="inner")

            # Определяем легитимные пакеты
            legit_mask = pd.Series(False, index=pkg_with_target.index)
            for prefix in LEGIT_PACKAGES_PREFIXES:
                legit_mask |= pkg_with_target["package"].str.startswith(prefix, na=False)

            # Только G-устройства (target=0) + легитимные пакеты + cert не NaN
            legit_g_certs = pkg_with_target[
                (pkg_with_target["target"] == 0) & legit_mask & pkg_with_target["cert"].notna()
                ]["cert"].unique()

            LEGIT_CERTS_GLOBAL = set(legit_g_certs)

            # Считаем долю легитимных сертификатов на устройство
            pkg_full["is_legit_cert"] = pkg_full["cert"].isin(LEGIT_CERTS_GLOBAL)
            n_legit_certs_new = pkg_full.groupby("agent_id")["is_legit_cert"].sum()
            n_total_pkgs = pkg_full.groupby("agent_id").size()
            frac_legit_new = (n_legit_certs_new / n_total_pkgs).fillna(0)

            # Добавляем в df
            df["frac_legit_certs"] = df["agent_id"].map(frac_legit_new).fillna(0)
        else:
            # Если нет target — не можем построить frac_legit_certs
            df["frac_legit_certs"] = 0.0

        # Заполняем NaN
        df["n_packages"] = df["n_packages"].fillna(0).astype(int)
        df["n_unique_certs"] = df["n_unique_certs"].fillna(0).astype(int)
        df["n_legit_apps"] = df["n_legit_apps"].fillna(0).astype(int)
        df["has_suspicious_pkg"] = df["has_suspicious_pkg"].fillna(0).astype(int)
        df["install_span_hours"] = df["install_span_hours"].fillna(0).astype(float)
        df["frac_legit_certs"] = df["frac_legit_certs"].fillna(0.0).astype(float)

    else:
        # Заглушки, если пакеты не используются
        df["n_packages"] = 0
        df["n_unique_certs"] = 0
        df["n_legit_apps"] = 0
        df["has_suspicious_pkg"] = 0
        df["install_span_hours"] = 0.0
        df["frac_legit_certs"] = 0.0

    # === Энтропия серийного номера ===
    if "Serial" in df.columns:
        df["serial_entropy"] = df["Serial"].apply(entropy)
    else:
        df["serial_entropy"] = 0.0

    # === Build-колонки и пропуски ===
    build_cols = [
        "PhoneHost", "AndroidSDK", "PhoneBootloader", "PhoneBoard",
        "PhoneProduct", "AndroidRelease", "PhoneFinterprint",
        "PhoneManufacturerModel", "PhoneID", "PhoneBrand",
        "PhoneDevice", "PhoneHardware", "Serial", "PhoneDisplay"
    ]

    for col in build_cols:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(int)

    existing_build_cols = [c for c in build_cols if c in df.columns]
    df["is_missing_build_info"] = df[existing_build_cols].isna().any(axis=1).astype(int)

    # === Бинаризация флагов ===
    flag_cols = ["NonMarketAppsEnabled", "IsDeviceSecured", "DeveloperModeEnabled"]
    for col in flag_cols:
        if col in df.columns:
            df[col] = df[col].apply(safe_bool_to_int)

    if "DebuggerConnected" in df.columns:
        df["DebuggerConnected"] = df["DebuggerConnected"].apply(parse_debugger)

    # === Специфические сигнатуры ===
    if "Serial" in df.columns:
        df["is_common_serial"] = (df["Serial"] == COMMON_SERIAL).astype(int)
    else:
        df["is_common_serial"] = 0

    if "PhoneDisplay" in df.columns or "PhoneFinterprint" in df.columns:
        display_test = df["PhoneDisplay"].apply(has_test_keys) if "PhoneDisplay" in df.columns else False
        fprint_test = df["PhoneFinterprint"].apply(has_test_keys) if "PhoneFinterprint" in df.columns else False
        df["is_test_keys"] = (display_test | fprint_test).astype(int)
    else:
        df["is_test_keys"] = 0

    if "PhoneHost" in df.columns:
        df["is_suspicious_host"] = df["PhoneHost"].fillna("").str.lower().apply(
            lambda x: int(any(pat in x for pat in SUSPICIOUS_HOST_PATTERNS))
        )
    else:
        df["is_suspicious_host"] = 0

    if "AndroidSDK" in df.columns:
        df["is_old_sdk"] = ((df["AndroidSDK"] < 28) & df["AndroidSDK"].notna()).astype(int)
    else:
        df["is_old_sdk"] = 0

    if "IsDeviceSecured" in df.columns:
        df["is_unsecured"] = (df["IsDeviceSecured"] == 0).astype(int)
    else:
        df["is_unsecured"] = 0

    # === Анализ отпечатков ===
    fingerprint_col = "PhoneFingerprint" if "PhoneFingerprint" in df.columns else "PhoneFinterprint"
    if fingerprint_col in df.columns:
        fprint_freq = df[fingerprint_col].map(df[fingerprint_col].value_counts())
        df["fprint_freq"] = fprint_freq.fillna(1)
        df["is_common_fprint"] = (df["fprint_freq"] > 100).astype(int)
        df["is_rare_fprint"] = (df["fprint_freq"] <= 1).astype(int)
        df["key_type"] = df[fingerprint_col].apply(extract_key_type)
        key_dummies = pd.get_dummies(df["key_type"], prefix="key")
        df = pd.concat([df, key_dummies], axis=1)
        df.drop(columns=["key_type"], inplace=True)

    # === Категориальные переменные: top-N + OTHER ===
    cat_cols = ["PhoneBrand", "PhoneHardware", "DefaultSMSApp"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("MISSING")
            top_cats = df[col].value_counts().head(30).index
            df[col] = df[col].where(df[col].isin(top_cats), "OTHER")

    # === Комбинированные признаки ===
    dev_mode = df["DeveloperModeEnabled"] if "DeveloperModeEnabled" in df.columns else pd.Series(0, index=df.index)
    df["risky_combo_1"] = (
            (df["NonMarketAppsEnabled"] == 1) &
            (df["IsDeviceSecured"] == 0) &
            (dev_mode == 1)
    ).astype(int)

    df["emulator_pattern"] = (
            (df["is_old_sdk"] == 1) &
            (df["DebuggerConnected"] == 1) &
            (df["is_common_serial"] == 1)
    ).astype(int)

    df["low_legit_ratio"] = (df["n_legit_apps"] / (df["n_packages"] + 1e-6)) < 0.1
    df["low_legit_ratio"] = df["low_legit_ratio"].astype(int)

    # === Удаление идентификаторов и чувствительных полей ===
    cols_to_drop = [
        "agent_id", "PhoneID", "PhoneFinterprint", "PhoneRadio", "Serial", "PhoneDisplay",
        "PhoneBootloader", "PhoneBoard", "PhoneProduct", "PhoneDevice",
        "PhoneManufacturerModel", "AndroidRelease", "PhoneHost", "PhoneFingerprint", "data"
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    return df