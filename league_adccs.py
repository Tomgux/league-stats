import os
import time
import random
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Regional routing: "europe", "americas", "asia", "sea"
# Platform routing: "euw1", "eun1", "na1", "kr", etc.
REGION_ROUTING = "europe"
PLATFORM_ROUTING = "euw1"

TARGET_MATCHES = 1000  
TIMESTAMPS = [10, 15, 20, 25]
OUTPUT_CSV = "silver_adc_cs_timelines.csv"
OUTPUT_PNG = "silver_adc_cs_per_min_winrate.png"


# ==============================================================================
# RIOT API DATA COLLECTOR (ADC ONLY)
# ==============================================================================
class RiotSilverADCCollector:
    def __init__(self, api_key: str, region: str, platform: str):
        self.headers = {"X-Riot-Token": api_key}
        self.region_url = f"https://{region}.api.riotgames.com"
        self.platform_url = f"https://{platform}.api.riotgames.com"

    def _get(self, url: str, params: dict = None) -> dict | list | None:
        while True:
            resp = requests.get(url, headers=self.headers, params=params, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 5))
                print(f"[Rate Limit] Sleeping {retry_after}s...")
                time.sleep(retry_after + 0.5)
            elif resp.status_code in (500, 502, 503, 504):
                time.sleep(2)
            else:
                if resp.status_code == 401:
                    raise RuntimeError(
                        "Riot API authentication failed (HTTP 401). Verify that "
                        "RIOT_API_KEY is current and valid, then try again."
                    )
                raise RuntimeError(
                    f"Riot API request failed with HTTP {resp.status_code} "
                    f"for {url}: {resp.text[:120]}"
                )

    def get_silver_puuids(self, players_per_division: int = 18) -> list[str]:
        puuids = set()
        for div in ["I", "II", "III", "IV"]:
            url = f"{self.platform_url}/lol/league/v4/entries/RANKED_SOLO_5x5/SILVER/{div}"
            entries = self._get(url, params={"page": random.randint(1, 5)}) or []
            random.shuffle(entries)

            added = 0
            for entry in entries:
                if added >= players_per_division:
                    break
                if entry.get("puuid"):
                    puuids.add(entry["puuid"])
                    added += 1
                elif entry.get("summonerId"):
                    summ = self._get(f"{self.platform_url}/lol/summoner/v4/summoners/{entry['summonerId']}")
                    if summ and summ.get("puuid"):
                        puuids.add(summ["puuid"])
                        added += 1
            print(f"Collected {added} PUUIDs from Silver {div}")
        return list(puuids)

    def get_ranked_match_ids(self, puuids: list[str], matches_per_player: int = 10) -> list[str]:
        match_ids = set()
        for puuid in puuids:
            url = f"{self.region_url}/lol/match/v5/matches/by-puuid/{puuid}/ids"
            ids = self._get(url, params={"queue": 420, "type": "ranked", "count": matches_per_player}) or []
            match_ids.update(ids)
        match_list = list(match_ids)
        random.shuffle(match_list)
        return match_list

    def extract_adc_rows(self, match_id: str, minutes: list[int]) -> list[dict]:
        """Extracts only BOTTOM (ADC) participants with CS/min and CS diff vs opposing ADC."""
        match_data = self._get(f"{self.region_url}/lol/match/v5/matches/{match_id}")
        if not match_data:
            return []

        info = match_data.get("info", {})
        if info.get("gameDuration", 0) < 300 or info.get("queueId") != 420:
            return []

        adcs = [p for p in info.get("participants", []) if p.get("teamPosition") == "BOTTOM"]
        if len(adcs) != 2:
            return []

        timeline_data = self._get(f"{self.region_url}/lol/match/v5/matches/{match_id}/timeline")
        if not timeline_data:
            return []

        frames = timeline_data.get("info", {}).get("frames", [])
        rows = []

        for idx, p in enumerate(adcs):
            opp = adcs[1 - idx]
            pid = str(p["participantId"])
            opp_pid = str(opp["participantId"])

            row = {
                "match_id": match_id,
                "champion": p.get("championName", ""),
                "win": int(p.get("win", False)),
            }

            for m in minutes:
                if m < len(frames):
                    p_frames = frames[m].get("participantFrames", {})
                    pf = p_frames.get(pid)
                    opf = p_frames.get(opp_pid)
                    if pf and opf:
                        my_cs = pf.get("minionsKilled", 0) + pf.get("jungleMinionsKilled", 0)
                        opp_cs = opf.get("minionsKilled", 0) + opf.get("jungleMinionsKilled", 0)
                        row[f"cspm_at_{m}"] = my_cs / m
                        row[f"csdiff_at_{m}"] = my_cs - opp_cs
                    else:
                        row[f"cspm_at_{m}"] = np.nan
                        row[f"csdiff_at_{m}"] = np.nan
                else:
                    row[f"cspm_at_{m}"] = np.nan
                    row[f"csdiff_at_{m}"] = np.nan

            rows.append(row)

        return rows


def collect_adc_dataset(api_key: str, target_matches: int) -> pd.DataFrame:
    collector = RiotSilverADCCollector(api_key, REGION_ROUTING, PLATFORM_ROUTING)
    puuids = collector.get_silver_puuids(players_per_division=18)
    match_ids = collector.get_ranked_match_ids(puuids, matches_per_player=10)[:target_matches]

    print(f"\nDownloading timelines for {len(match_ids)} Silver matches (ADC only)...")
    all_rows = []
    for idx, mid in enumerate(match_ids, 1):
        all_rows.extend(collector.extract_adc_rows(mid, TIMESTAMPS))
        if idx % 25 == 0:
            print(f"Progress: {idx}/{len(match_ids)} matches processed ({len(all_rows)} ADC games)...")

    return pd.DataFrame(all_rows)


# ==============================================================================
# PLOTTING ADC CS/MIN VS. WIN RATE
# ==============================================================================
def plot_adc_cspm_curves(df: pd.DataFrame, minutes: list[int]):
    if df.empty:
        print("No Silver ADC match data available; skipping chart.")
        return

    # Filter to BOTTOM if loading an older multi-role CSV
    if "role" in df.columns:
        df = df[df["role"] == "BOTTOM"].copy()

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=130)

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    x_grid = np.linspace(3.0, 10.5, 300).reshape(-1, 1)
    bin_edges = np.arange(3.0, 11.0, 0.5)  # 0.5 CS/min buckets

    for m, color in zip(minutes, colors):
        col = f"cspm_at_{m}"
        sub = df[[col, "win"]].dropna()
        if len(sub) < 40:
            print(f"Skipping {m}m (insufficient ongoing ADC rows: {len(sub)})")
            continue

        X = sub[[col]].values
        y = sub["win"].values

        clf = LogisticRegression()
        clf.fit(X, y)
        y_prob = clf.predict_proba(x_grid)[:, 1] * 100

        ax.plot(
            x_grid.flatten(),
            y_prob,
            color=color,
            linewidth=2.5,
            label=f"{m} min (n={len(sub):,})"
        )

        # Empirical 0.5 CS/min bin averages
        sub["bin"] = pd.cut(sub[col], bins=bin_edges)
        binned = sub.groupby("bin", observed=False)["win"].agg(["mean", "count"]).reset_index()
        binned = binned[binned["count"] >= 10]
        bin_centers = [interval.mid for interval in binned["bin"]]

        ax.scatter(
            bin_centers,
            binned["mean"] * 100,
            color=color,
            alpha=0.6,
            s=40
        )

    ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.8)

    ax.set_title(
        f"Silver ADC ({PLATFORM_ROUTING.upper()}): CS per Minute vs. Win Rate by Game Stage",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("ADC CS per Minute at Checkpoint (CS/min)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Win Rate (%)", fontsize=11, fontweight="bold")
    ax.set_xlim(3.0, 10.0)
    ax.set_ylim(20, 85)
    ax.legend(title="Game Stage (Ongoing Matches)", frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG)
    print(f"Chart saved to {OUTPUT_PNG}")
    plt.show()


# ==============================================================================
# RUN
# ==============================================================================
if __name__ == "__main__":
    # If you already collected the previous multi-role CSV, it reuses it automatically
    if os.path.exists("silver_soloq_cs_timelines.csv"):
        print("Found existing silver_soloq_cs_timelines.csv — filtering to ADC...")
        df_adc = pd.read_csv("silver_soloq_cs_timelines.csv")
    elif os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, encoding="utf-8-sig") as csv_file:
            has_csv_content = bool(csv_file.read().strip())
        if has_csv_content:
            print(f"Found existing {OUTPUT_CSV} — loading...")
            df_adc = pd.read_csv(OUTPUT_CSV)
        else:
            df_adc = None
    else:
        df_adc = None

    if df_adc is None:
        api_key = os.environ.get("RIOT_API_KEY", "").strip()
        if not api_key:
            raise SystemExit("Set the RIOT_API_KEY environment variable before running this script.")

        df_adc = collect_adc_dataset(api_key, target_matches=TARGET_MATCHES)
        if df_adc.empty:
            raise SystemExit("No Silver ADC match data was collected; no CSV or chart was saved.")
        df_adc.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved {len(df_adc)} Silver ADC rows to {OUTPUT_CSV}")

    plot_adc_cspm_curves(df_adc, TIMESTAMPS)