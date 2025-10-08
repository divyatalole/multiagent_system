import os
import argparse
import numpy as np
import pandas as pd

from models.startup_success_model import StartupSuccessModel, StartupFeatures


def generate_synthetic(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    sectors = ["fintech", "healthtech", "saas", "ecommerce", "ai", "edtech", "iot"]
    stages = ["idea", "seed", "series_a", "series_b"]
    regions = ["global", "us", "eu", "apac"]
    rows = []
    for _ in range(n):
        sector = rng.choice(sectors)
        stage = rng.choice(stages)
        region = rng.choice(regions)
        team_size = int(rng.integers(2, 80))
        competitiveness = int(rng.integers(2, 10))
        # Success heuristic: AI/SaaS + seed/series_a + team 8-30 + lower competitiveness
        base = 0.35
        if sector in {"ai", "saas"}: base += 0.15
        if stage in {"seed", "series_a"}: base += 0.1
        if 8 <= team_size <= 30: base += 0.1
        base += (10 - competitiveness) * 0.02
        y = 1 if rng.random() < min(max(base, 0.05), 0.95) else 0
        rows.append({
            "sector": sector,
            "team_size": team_size,
            "funding_stage": stage,
            "region": region,
            "market_competitiveness": competitiveness,
            "success": y,
        })
    return pd.DataFrame(rows)


def featurize_df(df: pd.DataFrame, model: StartupSuccessModel) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for _, r in df.iterrows():
        feats = StartupFeatures(
            sector=r["sector"],
            team_size=int(r["team_size"]),
            funding_stage=r["funding_stage"],
            region=r.get("region", "global"),
            market_competitiveness=int(r.get("market_competitiveness", 5)),
        )
        X.append(model._featurize(feats))
        y.append(int(r["success"]))
    return np.vstack(X), np.array(y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="", help="Path to historical startup CSV")
    parser.add_argument("--out", type=str, default="models/startup_success_rf.joblib", help="Output model path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    model = StartupSuccessModel(model_path=args.out)

    if args.csv and os.path.exists(args.csv):
        df = pd.read_csv(args.csv)
    else:
        df = generate_synthetic(800)

    X, y = featurize_df(df, model)
    model.train(X, y)
    print(f"Trained model saved to {args.out}. Dataset size: {len(df)}")


if __name__ == "__main__":
    main()




