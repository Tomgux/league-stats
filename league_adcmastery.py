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
REGION_ROUTING = "europe"     # "europe", "americas", "asia", "sea"
PLATFORM_ROUTING = "euw1"     # "euw1", "eun1", "na1", "kr", etc.

TARGET_MATCHES = 1000          # 350 matches = 700 Silver ADC rows
OUTPUT_CSV = "silver_adc_mastery.csv"
OUTPUT_PNG = "silver_adc_mastery_winrate.png"


# ==============================================================================
# RIOT API CLIENT (MATCH + CHAMPION MASTERY V4)
# ==============================================================================
class RiotMasteryCollector:
    def __init__(self, api_key: str, region: str, platform: str):
        self.headers = {"X-Riot-Token": api_key}
        self.region_url = f"https://{region}.api.riotgames.com"
        self.platform_url = f"https://{platform}.api.riotgames.com"
        self.mastery_cache: dict[tuple[str, int], tuple[int, int]] = {}

    def _get(self, url: str, params: dict = None, allow_404: bool = False) -> dict | list | None:
        while True:
            resp = requests.get(url, headers=self.headers, params=params, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            elif allow_404 and resp.status_code == 404:
                # 404 on mastery endpoint means 0 mastery points on that champion
                return {"championPoints": 0, "championLevel": 0}
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
                if resp.status_code == 403:
                    raise RuntimeError(
                        "Riot API access denied (HTTP 403). Your API key may be "
                        "expired or lack access to this endpoint."
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

    def get_champion_mastery(self, puuid: str, champion_id: int) -> tuple[int, int]:
        """Returns (championPoints, championLevel) with in-memory caching."""
        key = (puuid, champion_id)
        if key in self.mastery_cache:
            return self.mastery_cache[key]

        url = f"{self.platform_url}/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}/by-champion/{champion_id}"
        data = self._get(url, allow_404=True)
        if not data:
            return (0, 0)

        result = (int(data.get("championPoints", 0)), int(data.get("championLevel", 0)))
        self.mastery_cache[key] = result
        return result

    def extract_adc_mastery_rows(self, match_id: str) -> list[dict]:
        """Extracts both BOTTOM (ADC) participants and joins their Champion Mastery."""
        match_data = self._get(f"{self.region_url}/lol/match/v5/matches/{match_id}")
        if not match_data:
            return []

        info = match_data.get("info", {})
        if info.get("gameDuration", 0) < 300 or info.get("queueId") != 420:
            return []

        adcs = [p for p in info.get("participants", []) if p.get("teamPosition") == "BOTTOM"]
        if len(adcs) != 2:
            return []

        rows = []
        for p in adcs:
            puuid = p.get("puuid")
            champ_id = p.get("championId")
            if not puuid or not champ_id:
                continue

            points, level = self.get_champion_mastery(puuid, champ_id)
            rows.append({
                "match_id": match_id,
                "puuid": puuid,
                "champion": p.get("championName", ""),
                "champion_id": champ_id,
                "mastery_points": points,
                "mastery_level": level,
                "win": int(p.get("win", False)),
            })

        return rows


def collect_mastery_dataset(api_key: str, target_matches: int) -> pd.DataFrame:
    collector = RiotMasteryCollector(api_key, REGION_ROUTING, PLATFORM_ROUTING)
    puuids = collector.get_silver_puuids(players_per_division=18)
    match_ids = collector.get_ranked_match_ids(puuids, matches_per_player=10)[:target_matches]

    print(f"\nCollecting ADC mastery data across {len(match_ids)} Silver matches...")
    all_rows = []
    for idx, mid in enumerate(match_ids, 1):
        all_rows.extend(collector.extract_adc_mastery_rows(mid))
        if idx % 25 == 0:
            print(f"Progress: {idx}/{len(match_ids)} matches ({len(all_rows)} ADC games)...")

    return pd.DataFrame(all_rows)


# ==============================================================================
# PLOTTING MASTERY BUCKETS & LOG-MASTERY CURVE
# ==============================================================================
def plot_mastery_winrate(df: pd.DataFrame):
    if df.empty:
        print("No Silver ADC mastery data available; skipping chart.")
        return

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), dpi=130)

    # --- Panel 1: Discrete Mastery Point Buckets ---
    bins = [-1, 10_000, 30_000, 75_000, 150_000, 300_000, float("inf")]
    labels = ["0–10k", "10k–30k", "30k–75k", "75k–150k", "150k–300k", "300k+"]
    df["mastery_bucket"] = pd.cut(df["mastery_points"], bins=bins, labels=labels)

    summary = df.groupby("mastery_bucket", observed=False)["win"].agg(["mean", "count"]).reset_index()
    winrates = summary["mean"] * 100
    counts = summary["count"]

    bars = ax1.bar(summary["mastery_bucket"], winrates, color="#2b5c8f", alpha=0.88, width=0.6)
    ax1.axhline(50, color="gray", linestyle="--", linewidth=1)
    ax1.set_title("Win Rate by Mastery Bracket", fontsize=12.5, fontweight="bold")
    ax1.set_xlabel("Champion Mastery Points", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Win Rate (%)", fontsize=10.5, fontweight="bold")
    ax1.set_ylim(30, 70)

    for bar, wr, n in zip(bars, winrates, counts):
        if n > 0:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                wr + 1.0,
                f"{wr:.1f}%\n(n={n})",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    # --- Panel 2: Continuous Log10(Mastery Points) Logistic Curve ---
    df["log_mastery"] = np.log10(df["mastery_points"].clip(lower=1000))
    X = df[["log_mastery"]].values
    y = df["win"].values

    clf = LogisticRegression()
    clf.fit(X, y)

    x_grid = np.linspace(3.0, 6.3, 300).reshape(-1, 1)  # 1,000 pts to ~2,000,000 pts
    y_prob = clf.predict_proba(x_grid)[:, 1] * 100

    ax2.plot(10 ** x_grid.flatten(), y_prob, color="#d62728", linewidth=2.5, label="Logistic Fit")
    ax2.scatter(
        summary["mastery_bucket"].map({
            "0–10k": 5_000,
            "10k–30k": 20_000,
            "30k–75k": 50_000,
            "75k–150k": 110_000,
            "150k–300k": 220_000,
            "300k+": 500_000,
        }),
        winrates,
        color="#1f77b4",
        s=55,
        zorder=5,
        label="Bucket Mean",
    )

    ax2.set_xscale("log")
    ax2.axhline(50, color="gray", linestyle="--", linewidth=1)
    ax2.set_title("Continuous Win Rate Curve (Log Mastery Scale)", fontsize=12.5, fontweight="bold")
    ax2.set_xlabel("Champion Mastery Points (Log Scale)", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Win Rate (%)", fontsize=10.5, fontweight="bold")
    ax2.set_ylim(30, 70)
    ax2.legend(frameon=True, facecolor="white")

    fig.suptitle(
        f"Silver ADC ({PLATFORM_ROUTING.upper()}): Champion Mastery vs. Win Rate (n={len(df):,} ADC games)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, bbox_inches="tight")
    print(f"Chart saved to {OUTPUT_PNG}")
    plt.show()


# ==============================================================================
# RUN
# ==============================================================================
if __name__ == "__main__":
    df_mastery = None
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, encoding="utf-8-sig") as csv_file:
            has_csv_content = bool(csv_file.read().strip())
        if not has_csv_content:
            os.remove(OUTPUT_CSV)
        else:
            print(f"Loading existing {OUTPUT_CSV}...")
            df_mastery = pd.read_csv(OUTPUT_CSV)

    if df_mastery is None:
        api_key = os.environ.get("RIOT_API_KEY", "").strip()
        if not api_key:
            raise SystemExit("Set the RIOT_API_KEY environment variable before running this script.")

        df_mastery = collect_mastery_dataset(api_key, target_matches=TARGET_MATCHES)
        if df_mastery.empty:
            raise SystemExit("No Silver ADC mastery data was collected; no CSV or chart was saved.")
        df_mastery.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved {len(df_mastery)} rows to {OUTPUT_CSV}")

    plot_mastery_winrate(df_mastery)