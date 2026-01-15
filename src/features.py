# src/features.py
import pandas as pd
import numpy as np
import re
import json
from typing import Dict, Any, Optional

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
    match = re.search(r"(True|False)", str(x))
    return 1 if match and match.group(1) == "True" else 0


def has_test_keys(val) -> bool:
    return pd.notna(val) and "test-keys" in str(val).lower()


def extract_key_type(val: Any) -> str:
    if pd.isna(val):
        return "missing"
    s = str(val).lower()
    if "test-keys" in s:
        return "test"
    elif "release-keys" in s:
        return "release"
    else:
        return "unknown"


def process_packages(packages_path: str) -> pd.DataFrame:
    """
    Обрабатывает android_packages.csv.gz и возвращает агрегированные признаки по agent_id.
    Полностью совместима с оригинальной логикой, но значительно быстрее.
    """
    # Читаем только нужные колонки
    packages_df = pd.read_csv(
        packages_path,
        compression="gzip",
        usecols=["agent_id", "package", "data"]
    )

    # === Извлечение полей из JSON без json.loads ===
    # Используем регулярные выражения для безопасного и быстрого парсинга
    packages_df["cert"] = packages_df["data"].str.extract(r'"cert"\s*:\s*"([^"]*)"')
    packages_df["installed"] = packages_df["data"].str.extract(r'"installed"\s*:\s*"([^"]*)"')

    # --- Признак 1: количество пакетов ---
    n_packages = packages_df.groupby("agent_id").size()

    # --- Признак 2: количество уникальных сертификатов ---
    # cert может быть NaN — nunique() игнорирует NaN по умолчанию
    n_unique_certs = packages_df.groupby("agent_id")["cert"].nunique()

    # --- Признак 3: количество легитимных приложений ---
    is_legit = pd.Series(False, index=packages_df.index)
    for prefix in LEGIT_PACKAGES_PREFIXES:
        is_legit |= packages_df["package"].str.startswith(prefix, na=False)
    n_legit_apps = packages_df[is_legit].groupby("agent_id").size().reindex(n_packages.index, fill_value=0)

    # --- Признак 4: наличие подозрительных пакетов ---
    # В оригинале: объединяли все package в строку и искали подстроки
    # Это эквивалентно: есть ли хотя бы один пакет, содержащий подстроку?
    has_suspicious = pd.Series(False, index=packages_df.index)
    for substr in SUSPICIOUS_PACKAGES_SUBSTR:
        has_suspicious |= packages_df["package"].str.contains(substr, case=False, na=False)
    # Если хотя бы один пакет подозрительный → 1
    has_suspicious_pkg = (
        packages_df[has_suspicious]
        .groupby("agent_id")
        .size()
        .fillna(0)
        .clip(lower=0, upper=1)
        .astype(int)
        .reindex(n_packages.index, fill_value=0)
    )

    # --- Признак 5: разброс дат установки (в часах) ---
    packages_df["installed_dt"] = pd.to_datetime(packages_df["installed"], errors="coerce")
    min_time = packages_df.groupby("agent_id")["installed_dt"].min()
    max_time = packages_df.groupby("agent_id")["installed_dt"].max()
    time_diff_sec = (max_time - min_time).dt.total_seconds().fillna(0)
    install_span = (time_diff_sec / 3600.0).reindex(n_packages.index, fill_value=0.0)

    # Собираем всё в один DataFrame
    pkg_features = pd.DataFrame({
        "agent_id": n_packages.index,
        "n_packages": n_packages.values,
        "n_unique_certs": n_unique_certs.reindex(n_packages.index, fill_value=0).values,
        "n_legit_apps": n_legit_apps.values,
        "has_suspicious_pkg": has_suspicious_pkg.values,
        "install_span_hours": install_span.values,
    })

    return pkg_features


def preprocess_features(
    df: pd.DataFrame,
    packages_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Основная функция препроцессинга.
    Если указан packages_path — загружает и мержит признаки из пакетов.
    """
    df = df.copy()
    df["target"] = df["target"].map({"F": 1, "G": 0})

    # === Загрузка признаков из пакетов (если указан путь) ===
    if packages_path is not None:
        pkg_feats = process_packages(packages_path)
        df = df.merge(pkg_feats, on="agent_id", how="left")
        # Заполняем NaN разумными значениями
        df["n_packages"] = df["n_packages"].fillna(0).astype(int)
        df["n_unique_certs"] = df["n_unique_certs"].fillna(0).astype(int)
        df["n_legit_apps"] = df["n_legit_apps"].fillna(0).astype(int)
        df["has_suspicious_pkg"] = df["has_suspicious_pkg"].fillna(0).astype(int)
        df["install_span_hours"] = df["install_span_hours"].fillna(0).astype(float)
    else:
        # Заглушки, если пакеты не используются
        df["n_packages"] = 0
        df["n_unique_certs"] = 0
        df["n_legit_apps"] = 0
        df["has_suspicious_pkg"] = 0
        df["install_span_hours"] = 0.0

    # === Build-колонки ===
    build_cols = [
        "PhoneHost", "AndroidSDK", "PhoneBootloader", "PhoneBoard",
        "PhoneProduct", "AndroidRelease", "PhoneFinterprint",
        "PhoneManufacturerModel", "PhoneID", "PhoneBrand",
        "PhoneDevice", "PhoneHardware", "Serial", "PhoneDisplay"
    ]

    # Пропуски как сигнал
    for col in build_cols:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(int)

    df["is_missing_build_info"] = df[build_cols].isna().any(axis=1).astype(int)

    # === Бинаризация флагов ===
    flag_cols = ["NonMarketAppsEnabled", "IsDeviceSecured", "DeveloperModeEnabled"]
    for col in flag_cols:
        if col in df.columns:
            df[col] = df[col].apply(safe_bool_to_int)

    if "DebuggerConnected" in df.columns:
        df["DebuggerConnected"] = df["DebuggerConnected"].apply(parse_debugger)

    # === Специфические сигнатуры ===
    df["is_common_serial"] = (df["Serial"] == COMMON_SERIAL).astype(int)

    df["is_test_keys"] = (
        df["PhoneDisplay"].apply(has_test_keys) |
        df["PhoneFinterprint"].apply(has_test_keys)
    ).astype(int)

    df["is_suspicious_host"] = df["PhoneHost"].fillna("").str.lower().apply(
        lambda x: int(any(pat in x for pat in SUSPICIOUS_HOST_PATTERNS))
    )

    df["is_old_sdk"] = ((df["AndroidSDK"] < 28) & df["AndroidSDK"].notna()).astype(int)
    df["is_unsecured"] = (df["IsDeviceSecured"] == 0).astype(int)

    # === Анализ отпечатков ===
    if "PhoneFingerprint" in df.columns:
        fprint_freq = df["PhoneFingerprint"].map(df["PhoneFingerprint"].value_counts())
        df["fprint_freq"] = fprint_freq.fillna(1)
        df["is_common_fprint"] = (df["fprint_freq"] > 100).astype(int)
        df["is_rare_fprint"] = (df["fprint_freq"] <= 1).astype(int)
        df["key_type"] = df["PhoneFingerprint"].apply(extract_key_type)
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
    df["risky_combo_1"] = (
        (df["NonMarketAppsEnabled"] == 1) &
        (df["IsDeviceSecured"] == 0) &
        (df["DeveloperModeEnabled"] == 1)
    ).astype(int)

    df["emulator_pattern"] = (
        (df["is_old_sdk"] == 1) &
        (df["DebuggerConnected"] == 1) &
        (df["is_common_serial"] == 1)
    ).astype(int)

    df["low_legit_ratio"] = (df["n_legit_apps"] / (df["n_packages"] + 1e-6)) < 0.1
    df["low_legit_ratio"] = df["low_legit_ratio"].astype(int)

    # === Удаление исходных идентификаторов и чувствительных полей ===
    cols_to_drop = [
        "agent_id", "PhoneID", "PhoneFinterprint", "PhoneRadio", "Serial", "PhoneDisplay",
        "PhoneBootloader", "PhoneBoard", "PhoneProduct", "PhoneDevice",
        "PhoneManufacturerModel", "AndroidRelease", "PhoneHost", "PhoneFingerprint"
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    return df