import os
import joblib
from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier


@dataclass
class StartupFeatures:
    sector: str
    team_size: int
    funding_stage: str
    region: str = "global"
    market_competitiveness: int = 5  # 1-10


def _one_hot(value: str, categories: Dict[str, int]) -> np.ndarray:
    vec = np.zeros(len(categories), dtype=float)
    if value in categories:
        vec[categories[value]] = 1.0
    return vec


class StartupSuccessModel:
    def __init__(self, model_path: str = "models/startup_success_rf.joblib"):
        self.model_path = model_path
        self.model: Optional[RandomForestClassifier] = None
        # Simple vocabularies for one-hot encoding
        self.sectors = {
            "fintech": 0, "healthtech": 1, "saas": 2, "ecommerce": 3,
            "ai": 4, "edtech": 5, "iot": 6
        }
        self.stages = {"idea": 0, "seed": 1, "series_a": 2, "series_b": 3}
        self.regions = {"global": 0, "us": 1, "eu": 2, "apac": 3}

    def _featurize(self, features: StartupFeatures) -> np.ndarray:
        sector_vec = _one_hot(features.sector.lower(), self.sectors)
        stage_vec = _one_hot(features.funding_stage.lower(), self.stages)
        region_vec = _one_hot(features.region.lower(), self.regions)
        team_size_norm = np.array([min(max(features.team_size, 1), 500) / 500.0])
        competitiveness_norm = np.array([min(max(features.market_competitiveness, 1), 10) / 10.0])
        return np.concatenate([sector_vec, stage_vec, region_vec, team_size_norm, competitiveness_norm])

    def load(self) -> bool:
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            return True
        return False

    def save(self) -> None:
        if self.model is not None:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)

    def train(self, X: np.ndarray, y: np.ndarray, n_estimators: int = 200, random_state: int = 42) -> None:
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, class_weight="balanced")
        rf.fit(X, y)
        self.model = rf
        self.save()

    def predict_proba(self, features: StartupFeatures) -> float:
        if self.model is None:
            raise RuntimeError("Model not loaded or trained")
        x = self._featurize(features).reshape(1, -1)
        proba = self.model.predict_proba(x)[0, 1]
        return float(proba)

    def extract_features_from_text(self, topic: str, kb_hint: Optional[str] = None) -> StartupFeatures:
        t = topic.lower()
        # Naive heuristics to map topic text to features
        sector = "ai" if "ai" in t else "saas" if "saas" in t else "fintech" if "fintech" in t else "ecommerce"
        stage = "seed" if any(k in t for k in ["seed", "early"]) else "series_a" if "series a" in t else "idea"
        team_size = 8 if any(k in t for k in ["mvp", "prototype"]) else 20 if "scaling" in t else 5
        competitiveness = 8 if any(k in t for k in ["crowded", "competitive"]) else 5
        return StartupFeatures(
            sector=sector,
            team_size=team_size,
            funding_stage=stage,
            region="global",
            market_competitiveness=competitiveness,
        )




