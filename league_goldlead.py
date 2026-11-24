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
# Regional routing for Match-v5: "europe", "americas", "asia", "sea"
# Platform routing for League-v4: "euw1", "eun1", "na1", "kr", etc.
REGION_ROUTING = "europe"
PLATFORM_ROUTING = "euw1"

TARGET_MATCHES = 1000
TIMESTAMPS = [10, 15, 20, 25]
OUTPUT_CSV = "silver_soloq_gold_timelines.csv"
OUTPUT_PNG = "silver_gold_lead_winrate.png"


# ==============================================================================
# RIOT API CLIENT
# ==============================================================================
class RiotSilverCollector:
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

    def get_silver_puuids(self, players_per_division: int = 15) -> list[str]:
        """Samples Silver I-IV Solo/Duo players and extracts their PUUIDs."""
        puuids = set()
        divisions = ["I", "II", "III", "IV"]

        for div in divisions:
            url = f"{self.platform_url}/lol/league/v4/entries/RANKED_SOLO_5x5/SILVER/{div}"
            # Randomize page slightly so you don't always pull the exact same entries
            entries = self._get(url, params={"page": random.randint(1, 5)}) or []
            random.shuffle(entries)

            added = 0
            for entry in entries:
                if added >= players_per_division:
                    break

                # Newer League-v4 responses often include 'puuid' directly
                if "puuid" in entry and entry["puuid"]:
                    puuids.add(entry["puuid"])
                    added += 1
                elif "summonerId" in entry:
                    summ = self._get(f"{self.platform_url}/lol/summoner/v4/summoners/{entry['summonerId']}")
                    if summ and "puuid" in summ:
                        puuids.add(summ["puuid"])
                        added += 1

            print(f"Collected {added} PUUIDs from Silver {div}")

        return list(puuids)

    def get_ranked_match_ids(self, puuids: list[str], matches_per_player: int = 10) -> list[str]:
        """Fetches unique Ranked Solo/Duo match IDs (queue=420)."""
        match_ids = set()
        for puuid in puuids:
            url = f"{self.region_url}/lol/match/v5/matches/by-puuid/{puuid}/ids"
            ids = self._get(url, params={"queue": 420, "type": "ranked", "count": matches_per_player}) or []
            match_ids.update(ids)

        match_list = list(match_ids)
        random.shuffle(match_list)
        return match_list

    def extract_match_gold_diffs(self, match_id: str, minutes: list[int]) -> dict | None:
        """Extracts Blue team (1-5) gold lead and win outcome at target minute frames."""
        match_data = self._get(f"{self.region_url}/lol/match/v5/matches/{match_id}")
        if not match_data:
            return None

        info = match_data.get("info", {})
        # Skip remakes (< 5 minutes) and verify queueId is Ranked Solo/Duo (420)
        if info.get("gameDuration", 0) < 300 or info.get("queueId") != 420:
            return None

        blue_win = next((int(t["win"]) for t in info.get("teams", []) if t["teamId"] == 100), None)
        if blue_win is None:
            return None

        timeline_data = self._get(f"{self.region_url}/lol/match/v5/matches/{match_id}/timeline")
        if not timeline_data:
            return None

        frames = timeline_data.get("info", {}).get("frames", [])
        row = {"match_id": match_id, "win": blue_win}

        for m in minutes:
            if m < len(frames):
                p_frames = frames[m].get("participantFrames", {})
                blue_gold = sum(p_frames[str(pid)]["totalGold"] for pid in range(1, 6) if str(pid) in p_frames)
                red_gold = sum(p_frames[str(pid)]["totalGold"] for pid in range(6, 11) if str(pid) in p_frames)
                row[f"golddiffat{m}"] = blue_gold - red_gold
            else:
                row[f"golddiffat{m}"] = np.nan

        return row


def collect_silver_dataset(api_key: str, target_matches: int) -> pd.DataFrame:
    collector = RiotSilverCollector(api_key, REGION_ROUTING, PLATFORM_ROUTING)
    puuids = collector.get_silver_puuids(players_per_division=15)
    match_ids = collector.get_ranked_match_ids(puuids, matches_per_player=10)[:target_matches]

    print(f"\nDownloading timelines for {len(match_ids)} Silver matches...")
    rows = []
    for idx, mid in enumerate(match_ids, 1):
        row = collector.extract_match_gold_diffs(mid, TIMESTAMPS)
        if row:
            rows.append(row)
        if idx % 20 == 0:
            print(f"Progress: {idx}/{len(match_ids)} matches processed ({len(rows)} valid)...")

    columns = ["match_id", "win", *(f"golddiffat{minute}" for minute in TIMESTAMPS)]
    return pd.DataFrame(rows, columns=columns)


# ==============================================================================
# PLOTTING & LOGISTIC MODEL
# ==============================================================================
def plot_silver_curves(df: pd.DataFrame, minutes: list[int]):
    if df.empty:
        print("No Silver match data available; skipping chart.")
        return

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=130)

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    x_grid = np.linspace(-8000, 8000, 400).reshape(-1, 1)
    bin_edges = np.arange(-8000, 8500, 1000)

    for m, color in zip(minutes, colors):
        col = f"golddiffat{m}"
        sub = df[[col, "win"]].dropna()
        if len(sub) < 25:
            print(f"Skipping {m}m (only {len(sub)} ongoing games)")
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

        sub["bin"] = pd.cut(sub[col], bins=bin_edges)
        binned = sub.groupby("bin", observed=False)["win"].agg(["mean", "count"]).reset_index()
        binned = binned[binned["count"] >= 5]
        bin_centers = [interval.mid for interval in binned["bin"]]

        ax.scatter(
            bin_centers,
            binned["mean"] * 100,
            color=color,
            alpha=0.55,
            s=35
        )

    ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.8)

    ax.set_title(f"Silver Solo Queue ({PLATFORM_ROUTING.upper()}): Team Gold Lead vs. Win Rate", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Team Gold Difference at Timestamp (Gold)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Win Rate (%)", fontsize=11, fontweight="bold")
    ax.set_xlim(-8000, 8000)
    ax.set_ylim(0, 100)
    ax.legend(title="Game Stage (Ongoing Matches)", frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG)
    print(f"Chart saved to {OUTPUT_PNG}")
    plt.show()


# ==============================================================================
# RUN
# ==============================================================================
if __name__ == "__main__":
    api_key = os.environ.get("RIOT_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set the RIOT_API_KEY environment variable before running this script.")

    df_silver = collect_silver_dataset(api_key, target_matches=TARGET_MATCHES)
    df_silver.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df_silver)} matches to {OUTPUT_CSV}")
    plot_silver_curves(df_silver, TIMESTAMPS)