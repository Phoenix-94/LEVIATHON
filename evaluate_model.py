from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Dummy test labels and predictions (Replace with actual test data)
y_true = ["Shark", "Dolphin", "Whale", "Jellyfish", "Shark", "Dolphin"]  # Actual labels
y_pred = ["Shark", "Dolphin", "Jellyfish", "Jellyfish", "Whale", "Dolphin"]  # Predicted labels

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=["Shark", "Dolphin", "Whale", "Jellyfish"])

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Shark", "Dolphin", "Whale", "Jellyfish"], 
            yticklabels=["Shark", "Dolphin", "Whale", "Jellyfish"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Print classification report
print(classification_report(y_true, y_pred, target_names=["Shark", "Dolphin", "Whale", "Jellyfish"]))
