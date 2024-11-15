from naive_bayes import NaiveBayesClassifier
import pandas as pd

DATASET_FILE: str = "play_tennis.csv"

classifier: NaiveBayesClassifier = NaiveBayesClassifier()
classifier.train(DATASET_FILE, target="PlayTennis")
classifier.save_model("naive_bayes_model.json")

df: pd.DataFrame = pd.read_csv(DATASET_FILE)
correct: int = 0

for i in range(len(df)):
    test_instance: dict[str, str] = df.iloc[i].drop("PlayTennis").to_dict()
    actual_label: str = df.iloc[i]["PlayTennis"]
    train_df: pd.DataFrame = df.drop(index=i)
    train_df.to_csv("temp_train.csv", index=False)
    classifier.train("temp_train.csv", target="PlayTennis")

    predicted_class, _ = classifier.predict(test_instance)
    print(
        f"Test Instance: {test_instance}, Actual: {actual_label}, Predicted: {predicted_class}"
    )

    if predicted_class == actual_label:
        correct += 1

accuracy: float = correct / len(df)
print(f"Accuracy: {round(accuracy * 100, 2)}%")
