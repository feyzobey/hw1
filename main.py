from naive_bayes import NaiveBayesClassifier

TRAIN_SET = "play_tennis.csv"
TEST_SET = "play_tennis_test.csv"
INSTANCE_TO_CLASSIFY = {
    "Outlook": "Sunny",
    "Temperature": "Cool",
    "Humidity": "Normal",
    "Wind": "Weak",
}

classifier = NaiveBayesClassifier()

classifier.train(TRAIN_SET, target="PlayTennis")

predicted_label, probabilities = classifier.classify_instance(INSTANCE_TO_CLASSIFY)
print(f"Instance to Classify: {INSTANCE_TO_CLASSIFY}")
print(f"Predicted Class: {predicted_label}")
print(f"Posterior Probabilities: {probabilities}")

# Classify the test set
classifier.classify_test_set(TEST_SET, target="PlayTennis")
