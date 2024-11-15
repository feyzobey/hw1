import pandas as pd
import json
from typing import Dict, Any, Tuple


class NaiveBayesClassifier:
    def __init__(self) -> None:
        self.priors: Dict[str, float] = {}
        self.likelihoods: Dict[str, Dict[str, Dict[Any, float]]] = {}
        self.classes: set[str] = set()

    def train(self, file_path: str, target: str) -> None:
        df: pd.DataFrame = pd.read_csv(file_path)

        self.classes = set(df[target].unique())
        total_instances: int = len(df)

        self.priors = {
            cls: len(df[df[target] == cls]) / total_instances for cls in self.classes
        }

        self.likelihoods = {
            feature: {cls: {} for cls in self.classes}
            for feature in df.columns
            if feature != target
        }

        for feature in df.columns:
            if feature == target:
                continue
            unique_values: int = df[feature].nunique()

            for cls in self.classes:
                subset: pd.DataFrame = df[df[target] == cls]
                total_count: int = len(subset)

                value_counts: pd.Series = subset[feature].value_counts()
                for value in df[feature].unique():
                    count: int = value_counts.get(value, 0)
                    self.likelihoods[feature][cls][value] = (count + 1) / (
                        total_count + unique_values
                    )

    def predict(self, instance: Dict[str, Any]) -> Tuple[str, Dict[str, float]]:
        posteriors: Dict[str, float] = {}
        for cls in self.classes:
            posterior: float = self.priors[cls]

            for feature, value in instance.items():
                if (
                    feature in self.likelihoods
                    and value in self.likelihoods[feature][cls]
                ):
                    posterior *= self.likelihoods[feature][cls][value]
                else:
                    unique_values: int = len(self.likelihoods[feature][cls])
                    posterior *= 1 / (
                        sum(self.likelihoods[feature][cls].values()) + unique_values
                    )

            posteriors[cls] = posterior

        return max(posteriors, key=posteriors.get), posteriors

    def save_model(self, filename: str) -> None:
        model: Dict[str, Any] = {
            "priors": self.priors,
            "likelihoods": self.likelihoods,
            "classes": list(self.classes),
        }
        with open(filename, "w") as f:
            json.dump(model, f)

    def load_model(self, filename: str) -> None:
        with open(filename, "r") as f:
            model: Dict[str, Any] = json.load(f)
        self.priors = model["priors"]
        self.likelihoods = model["likelihoods"]
        self.classes = set(model["classes"])
