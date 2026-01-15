# src/build_final_dataset.py
import pandas as pd
import numpy as np
import re
from pathlib import Path

# === КОНСТАНТЫ ===
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


# === Вспомогательные функции ===
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
    if "true" in s:
        return 1
    return 0

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


# === Быстрая обработка пакетов ===
def process_packages_fast(packages_path: str) -> pd.DataFrame:
    print("📦 Обрабатываем android_packages.csv.gz...")
    df = pd.read_csv(packages_path, compression="gzip", usecols=["agent_id", "package", "data"])

    # Извлекаем поля из JSON регулярками
    df["cert"] = df["data"].str.extract(r'"cert"\s*:\s*"([^"]*)"')
    df["installed"] = df["data"].str.extract(r'"installed"\s*:\s*"([^"]*)"')

    n_packages = df.groupby("agent_id").size()
    n_unique_certs = df.groupby("agent_id")["cert"].nunique()

    is_legit = pd.Series(False, index=df.index)
    for prefix in LEGIT_PACKAGES_PREFIXES:
        is_legit |= df["package"].str.startswith(prefix, na=False)
    n_legit_apps = df[is_legit].groupby("agent_id").size().reindex(n_packages.index, fill_value=0)

    is_susp = pd.Series(False, index=df.index)
    for substr in SUSPICIOUS_PACKAGES_SUBSTR:
        is_susp |= df["package"].str.contains(substr, case=False, na=False)
    has_suspicious_pkg = (
        df[is_susp].groupby("agent_id").size()
        .fillna(0).clip(upper=1).astype(int)
        .reindex(n_packages.index, fill_value=0)
    )

    # --- Временные признаки ---
    df["installed_dt"] = pd.to_datetime(df["installed"], errors="coerce", unit="ms")  # ⚠️ важно: unit="ms"!
    min_t = df.groupby("agent_id")["installed_dt"].min()
    max_t = df.groupby("agent_id")["installed_dt"].max()
    span = (max_t - min_t).dt.total_seconds().fillna(0) / 3600.0
    install_span = span.reindex(n_packages.index, fill_value=0.0)

    return pd.DataFrame({
        "agent_id": n_packages.index,
        "n_packages": n_packages.values,
        "n_unique_certs": n_unique_certs.reindex(n_packages.index, fill_value=0).values,
        "n_legit_apps": n_legit_apps.values,
        "has_suspicious_pkg": has_suspicious_pkg.values,
        "install_span_hours": install_span.values,
    })


# === Основная функция ===
def build_final_dataset(
    device_info_path: str,
    packages_path: str,
    output_path: str = "data/final_dataset.csv",
    force_rebuild: bool = False
):
    output_path = Path(output_path)
    if output_path.exists() and not force_rebuild:
        print(f"✅ Финальный датасет уже существует: {output_path}")
        return

    print("📥 Загружаем android_device_info.csv...")
    df = pd.read_csv(device_info_path)

    # Целевая переменная
    df["target"] = df["target"].map({"F": 1, "G": 0})

    # Обрабатываем пакеты
    pkg_feats = process_packages_fast(packages_path)
    df = df.merge(pkg_feats, on="agent_id", how="left")

    if "target" in df.columns:
        # Загружаем пакеты ещё раз (или сохрани их ранее)
        pkg_full = pd.read_csv(packages_path, compression="gzip", usecols=["agent_id", "package", "data"])
        pkg_full["cert"] = pkg_full["data"].str.extract(r'"cert"\s*:\s*"([^"]*)"')

        # Мержим с target
        pkg_with_target = pkg_full.merge(df[["agent_id", "target"]], on="agent_id", how="inner")

        # Берём только G (target=0) и legit-пакеты
        legit_mask = pd.Series(False, index=pkg_with_target.index)
        for prefix in LEGIT_PACKAGES_PREFIXES:
            legit_mask |= pkg_with_target["package"].str.startswith(prefix, na=False)

        legit_g_certs = pkg_with_target[
            (pkg_with_target["target"] == 0) & legit_mask & pkg_with_target["cert"].notna()
            ]["cert"].unique()

        LEGIT_CERTS_GLOBAL = set(legit_g_certs)

        # Теперь пересчитываем frac_legit_certs
        pkg_full["is_legit_cert"] = pkg_full["cert"].isin(LEGIT_CERTS_GLOBAL)
        n_legit_certs_new = pkg_full.groupby("agent_id")["is_legit_cert"].sum()
        n_total_pkgs = pkg_full.groupby("agent_id").size()
        frac_legit_new = (n_legit_certs_new / n_total_pkgs).fillna(0)

        # Обновляем в основном df
        df["frac_legit_certs"] = df["agent_id"].map(frac_legit_new).fillna(0)

    from collections import Counter
    import math

    def entropy(s):
        if not s or pd.isna(s):
            return 0.0
        p, lns = Counter(s), float(len(s))
        return -sum(count / lns * math.log(count / lns + 1e-9) for count in p.values())

    df["serial_entropy"] = df["Serial"].apply(entropy)


    # Заполняем NaN
    df["n_packages"] = df["n_packages"].fillna(0).astype(int)
    df["n_unique_certs"] = df["n_unique_certs"].fillna(0).astype(int)
    df["n_legit_apps"] = df["n_legit_apps"].fillna(0).astype(int)
    df["has_suspicious_pkg"] = df["has_suspicious_pkg"].fillna(0).astype(int)
    df["install_span_hours"] = df["install_span_hours"].fillna(0).astype(float)

    # === Build-колонки ===
    build_cols = [
        "PhoneHost", "AndroidSDK", "PhoneBootloader", "PhoneBoard",
        "PhoneProduct", "AndroidRelease", "PhoneFinterprint",
        "PhoneManufacturerModel", "PhoneID", "PhoneBrand",
        "PhoneDevice", "PhoneHardware", "Serial", "PhoneDisplay"
    ]

    for col in build_cols:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(int)
    df["is_missing_build_info"] = df[[c for c in build_cols if c in df.columns]].isna().any(axis=1).astype(int)

    # === Флаги ===
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
    if "PhoneFinterprint" in df.columns:
        fprint_freq = df["PhoneFinterprint"].map(df["PhoneFinterprint"].value_counts())
        df["fprint_freq"] = fprint_freq.fillna(1)
        df["is_common_fprint"] = (df["fprint_freq"] > 100).astype(int)
        df["is_rare_fprint"] = (df["fprint_freq"] <= 1).astype(int)
        df["key_type"] = df["PhoneFinterprint"].apply(extract_key_type)
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
        (df.get("DeveloperModeEnabled", pd.Series(0, index=df.index)) == 1)
    ).astype(int)

    df["emulator_pattern"] = (
        (df["is_old_sdk"] == 1) &
        (df["DebuggerConnected"] == 1) &
        (df["is_common_serial"] == 1)
    ).astype(int)

    df["low_legit_ratio"] = (df["n_legit_apps"] / (df["n_packages"] + 1e-6)) < 0.1
    df["low_legit_ratio"] = df["low_legit_ratio"].astype(int)

    # === Удаляем идентификаторы и чувствительные поля ===
    cols_to_drop = [
        "agent_id", "PhoneID", "PhoneFinterprint", "PhoneRadio", "Serial", "PhoneDisplay",
        "PhoneBootloader", "PhoneBoard", "PhoneProduct", "PhoneDevice",
        "PhoneManufacturerModel", "AndroidRelease", "PhoneHost", "PhoneFingerprint", "data"
    ]

    df["has_mixed_keys"] = (
            df["PhoneFinterprint"].str.contains("test-keys", na=False) &
            df["PhoneFinterprint"].str.contains("release-keys", na=False)
    ).astype(int)

    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    # Сохраняем в CSV
    print(f"💾 Сохраняем финальный датасет в {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Готово! Размер: {df.shape}, признаков: {df.shape[1] - 1}")

if __name__ == "__main__":
    build_final_dataset(
        device_info_path=r"D:\Code\test_task_f6\test_task_f6\train_dataset.csv",
        packages_path=r"D:\Code\test_task_f6\test_task_f6\android_packages.csv.gz",
        output_path=r"D:\Code\test_task_f6\test_task_f6\train_dataset_6.csv",
        force_rebuild=False
    )