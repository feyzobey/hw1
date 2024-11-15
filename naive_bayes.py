import pandas as pd
import numpy as np
import json
from typing import Dict, Any, Tuple


class NaiveBayesClassifier:
    def __init__(self):
        self.priors: Dict[str, float] = {}
        self.likelihoods: Dict[str, Dict[str, Dict[Any, float]]] = {}
        self.classes: set[str] = set()
        self.model_file: str = "naive_bayes_model.json"
        self.log_file: str = "naive_bayes_log.txt"

    def train(self, dataset_path: str, target: str) -> None:
        df: pd.DataFrame = pd.read_csv(dataset_path)
        total_instances: int = len(df)

        self.classes = set(df[target].unique())
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
            for cls in self.classes:
                subset = df[df[target] == cls]
                value_counts = subset[feature].value_counts()
                unique_values = df[feature].nunique()
                total_count = len(subset)

                self.likelihoods[feature][cls] = {
                    val: (value_counts.get(val, 0) + 1) / (total_count + unique_values)
                    for val in df[feature].unique()
                }

        # Save the trained model
        self._save_model()

    def classify_instance(
        self, instance: Dict[str, Any]
    ) -> Tuple[str, Dict[str, float]]:
        self._load_model()

        posteriors: Dict[str, float] = {}
        log_likelihoods: Dict[str, float] = {
            cls: np.log(self.priors[cls]) for cls in self.classes
        }

        for cls in self.classes:
            posterior: float = np.log(self.priors[cls])

            for feature, value in instance.items():
                if (
                    feature in self.likelihoods
                    and value in self.likelihoods[feature][cls]
                ):
                    likelihood: float = self.likelihoods[feature][cls][value]
                else:
                    unique_values: int = len(self.likelihoods[feature][cls])
                    likelihood: float = 1 / (
                        sum(self.likelihoods[feature][cls].values()) + unique_values
                    )
                posterior += np.log(likelihood)

            posteriors[cls] = np.exp(posterior)
            log_likelihoods[cls] = posterior

        total: float = sum(posteriors.values())
        for cls in posteriors:
            posteriors[cls] /= total

        # Log detailed calculations
        self._log_calculation(instance, log_likelihoods)

        # Return the class with the highest posterior probability
        return max(posteriors, key=posteriors.get), posteriors

    def classify_test_set(self, test_set_path: str, target: str) -> None:
        df: pd.DataFrame = pd.read_csv(test_set_path)
        correct: int = 0

        with open(self.log_file, "w") as log_file:
            log_file.write("--------------------------------------------------\n")
            log_file.write("| Instance |   Actual   | Predicted  | Correct  |\n")
            log_file.write("--------------------------------------------------\n")

            for i, row in df.iterrows():
                instance: Dict[str, Any] = row.drop(target).to_dict()
                actual_label: Any = row[target]
                predicted_label, _ = self.classify_instance(instance)
                is_correct: bool = predicted_label == actual_label
                correct += is_correct

                log_file.write(
                    f"|    {i+1:<5} |    {actual_label:<8} |    {predicted_label:<8} |   {'True' if is_correct else 'False':<5}   |\n"
                )

            accuracy: float = correct / len(df)
            log_file.write("--------------------------------------------------\n")
            log_file.write(f"Overall Accuracy: {accuracy:.2f}\n")

    def _save_model(self) -> None:
        model: Dict[str, Any] = {
            "priors": self.priors,
            "likelihoods": self.likelihoods,
            "classes": list(self.classes),
        }
        with open(self.model_file, "w") as f:
            json.dump(model, f)

    def _load_model(self) -> None:
        with open(self.model_file, "r") as f:
            model: Dict[str, Any] = json.load(f)
        self.priors = model["priors"]
        self.likelihoods = model["likelihoods"]
        self.classes = set(model["classes"])

    def _log_calculation(
        self, instance: Dict[str, Any], log_likelihoods: Dict[str, float]
    ) -> None:
        with open("naive_bayes_calculations.txt", "a") as f:
            f.write(f"--- Making Prediction ---\n")
            f.write(f"Instance to predict: {instance}\n")
            for cls, log_prob in log_likelihoods.items():
                f.write(
                    f"Initial score for class '{cls}' (Log(P({cls}))) = {np.log(self.priors[cls]):.6f}\n"
                )

            for feature, value in instance.items():
                f.write(f"Processing feature: {feature} | Value: {value}\n")
                for cls in self.classes:
                    likelihood: float = self.likelihoods[feature][cls].get(
                        value,
                        1
                        / (
                            sum(self.likelihoods[feature][cls].values())
                            + len(self.likelihoods[feature][cls])
                        ),
                    )
                    f.write(f"  P({feature}={value} | {cls}) = {likelihood:.6f}\n")
                    f.write(
                        f"  Updated score for class '{cls}' = {log_likelihoods[cls]:.6f}\n"
                    )

            f.write("\nFinal Scores:\n")
            for cls, log_prob in log_likelihoods.items():
                f.write(f"  Class '{cls}': {log_prob:.6f}\n")
