"""Fetch semua sirkuit 2023-2026 dari OpenF1 -> data/raw/.

Dibuat supaya fetch panjang bisa jalan di background dan bisa dilanjutkan
(cache per session_key tidak diambil ulang). Logika sama dengan 00_fetch.ipynb.

Jalankan:
    ./venvf1/Scripts/python.exe scripts/fetch_all.py
"""

import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent

with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}

RAW_DIR = ROOT / cfg["paths"]["raw"]

# Hanya sesi pre-qualifying (Race tidak dipakai lagi untuk prediksi kualifikasi).
SESSION_NAMES = {"Practice 1", "Practice 2", "Practice 3", "Qualifying"}
PRACTICE_ENDPOINTS = ("laps", "stints", "weather", "drivers")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
LOGGER = logging.getLogger("fetch_all")


class OpenF1AccessError(RuntimeError):
    pass


class OpenF1Client:
    def __init__(self, base_url, timeout, max_retries, rate_limit_per_minute):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_interval = 60.0 / max(rate_limit_per_minute, 1)
        self._last_request_at = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "prediksi-openf1/0.3"})

    def _throttle(self):
        wait = self.min_interval - (time.monotonic() - self._last_request_at)
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.monotonic()

    def get(self, endpoint, **params):
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)

                if response.status_code in (401, 403):
                    try:
                        detail = response.json().get("detail", response.text)
                    except Exception:
                        detail = response.text
                    raise OpenF1AccessError(
                        f"OpenF1 menolak request ({response.status_code}). "
                        "Biasanya ada sesi live; akses historis dibatasi sampai selesai. "
                        f"Detail: {detail}"
                    )

                if response.status_code == 404:
                    LOGGER.warning("Tidak ada data untuk %s params=%s", endpoint, params)
                    return []

                if response.status_code == 429:
                    wait = min(60, 2**attempt)
                    LOGGER.warning("Rate limit tercapai, tunggu %ss", wait)
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict) and "detail" in payload:
                    raise RuntimeError(f"API error: {payload['detail']}")
                return payload

            except OpenF1AccessError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                wait = 2**attempt
                LOGGER.warning("Request gagal (%s), retry dalam %ss", exc, wait)
                time.sleep(wait)

        raise RuntimeError(f"Gagal mengambil {url} dengan params={params}") from last_error

    def get_sessions(self, years):
        sessions = []
        for year in years:
            payload = self.get("sessions", year=year)
            if isinstance(payload, list):
                sessions.extend(payload)
        return sessions


# ---- Cache ----

def cache_path(endpoint, key, suffix=".parquet"):
    return RAW_DIR / endpoint / f"{key}{suffix}"


def to_dataframe(payload):
    if payload is None:
        return pd.DataFrame()
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict):
        return pd.DataFrame([payload])
    return pd.DataFrame(payload)


def load_cached(endpoint, key):
    parquet_path = cache_path(endpoint, key)
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    json_path = cache_path(endpoint, key, ".json")
    if json_path.exists():
        return pd.read_json(json_path)
    return None


def save_cached(df, endpoint, key):
    path = cache_path(endpoint, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception:
        fallback = path.with_suffix(".json")
        df.to_json(fallback, orient="records", date_format="iso")
        return fallback


def cached_get(client, endpoint, key, *, force=False, **params):
    if not force:
        cached = load_cached(endpoint, key)
        if cached is not None:
            return cached, True
    payload = client.get(endpoint, **params)
    df = to_dataframe(payload)
    save_cached(df, endpoint, key)
    return df, False


# ---- Fetch ----

def select_sessions(sessions):
    return [
        s for s in sessions
        if s.get("session_name") in SESSION_NAMES and not s.get("is_cancelled", False)
    ]


def group_by_meeting(sessions):
    grouped = {}
    for item in sessions:
        grouped.setdefault(item["meeting_key"], []).append(item)
    return {
        key: sorted(items, key=lambda item: item.get("date_start", ""))
        for key, items in grouped.items()
    }


def fetch_weekend(client, sessions):
    records = []
    qualifying = next(
        (s for s in sessions if s.get("session_name") == "Qualifying"), None
    )
    for session in sessions:
        session_key = session["session_key"]
        session_name = session.get("session_name", "")

        for endpoint in PRACTICE_ENDPOINTS:
            df, was_cached = cached_get(
                client, endpoint, f"session_{session_key}", session_key=session_key
            )
            records.append({
                "session_key": session_key,
                "session_name": session_name,
                "endpoint": endpoint,
                "rows": len(df),
                "cached": was_cached,
            })

        # starting_grid diambil dari sesi Qualifying, bukan Race.
        if session_name == "Qualifying":
            if qualifying is None:
                LOGGER.warning("Qualifying tidak ditemukan, skip starting_grid.")
                continue
            df, was_cached = cached_get(
                client,
                "starting_grid",
                f"session_{session_key}",
                session_key=session_key,
            )
            records.append({
                "session_key": session_key,
                "session_name": "Qualifying",
                "endpoint": "starting_grid",
                "rows": len(df),
                "cached": was_cached,
            })
    return records


def main():
    api_cfg = cfg.get("api", {})
    data_cfg = cfg.get("data", {})
    years = list(data_cfg.get("years", [2023, 2024, 2025, 2026]))

    client = OpenF1Client(
        base_url=api_cfg.get("base_url", "https://api.openf1.org/v1"),
        timeout=api_cfg.get("timeout_seconds", 30),
        max_retries=api_cfg.get("max_retries", 3),
        rate_limit_per_minute=api_cfg.get("rate_limit_per_minute", 30),
    )

    LOGGER.info("Ambil daftar sesi untuk tahun %s ...", years)
    sessions = client.get_sessions(years)
    selected = select_sessions(sessions)
    weekends = group_by_meeting(selected)

    save_cached(pd.DataFrame(selected), "sessions", "all_circuits")

    LOGGER.info(
        "Ditemukan %s weekend, %s sesi. Mulai fetch (cache lama dilewati).",
        len(weekends), len(selected),
    )

    all_records = []
    total = len(weekends)
    for index, (meeting_key, weekend_sessions) in enumerate(weekends.items(), start=1):
        circuit = weekend_sessions[0].get("circuit_short_name", "?")
        year = weekend_sessions[0].get("year", "?")
        LOGGER.info(
            "[%s/%s] %s %s | meeting_key=%s",
            index, total, year, circuit, meeting_key,
        )
        try:
            all_records.extend(fetch_weekend(client, weekend_sessions))
        except OpenF1AccessError as exc:
            LOGGER.error("Dibatalkan oleh OpenF1: %s", exc)
            LOGGER.error("Cache sejauh ini tersimpan. Jalankan ulang nanti untuk melanjutkan.")
            pd.DataFrame(all_records).to_csv(RAW_DIR / "_fetch_log.csv", index=False)
            return 1

    summary = pd.DataFrame(all_records)
    summary.to_csv(RAW_DIR / "_fetch_log.csv", index=False)
    cached_count = int(summary["cached"].sum()) if len(summary) else 0
    LOGGER.info(
        "Selesai. %s request (%s dari cache, %s baru). Log: %s",
        len(summary), cached_count, len(summary) - cached_count,
        RAW_DIR / "_fetch_log.csv",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
