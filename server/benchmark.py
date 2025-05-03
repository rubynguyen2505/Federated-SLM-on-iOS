import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import ast
plt.ion()

# Load Federated Learning metrics
federated_df = pd.read_csv("federated_metrics_log.csv")  # Use the actual filename

# Load Centralized model metrics
centralized_df = pd.read_csv("centralized_model_metrics.csv")

# Get the latest centralized metrics (last row if you logged multiple)
centralized_metrics = centralized_df.iloc[-1]

# Create a centralized metrics DataFrame to match federated format
centralized_over_rounds = pd.DataFrame({
    "round": federated_df["round"],
    "accuracy": [centralized_metrics["accuracy"]] * len(federated_df),
    "precision": [centralized_metrics["precision"]] * len(federated_df),
    "recall": [centralized_metrics["recall"]] * len(federated_df),
    "f1_score": [centralized_metrics["f1_score"]] * len(federated_df),
    "model_type": ["centralized"] * len(federated_df)
})

# Add model_type column to federated metrics
federated_df["model_type"] = "federated"

# Combine both into one DataFrame
combined_df = pd.concat([federated_df[["round", "accuracy", "precision", "recall", "f1_score", "model_type"]],
                         centralized_over_rounds], ignore_index=True)

# Plotting
metrics = ["accuracy", "precision", "recall", "f1_score"]
for metric in metrics:
    plt.figure(figsize=(8, 5))
    for model_type in combined_df["model_type"].unique():
        data = combined_df[combined_df["model_type"] == model_type]
        plt.plot(data["round"], data[metric], label=model_type.capitalize(), marker='o' if model_type == "federated" else 'd')
    plt.title(f"{metric.capitalize()} Over Rounds")
    plt.xlabel("Round")
    plt.ylabel(metric.capitalize())
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.ylim(0, 1)
    plt.savefig(f"{metric}_comparison.png")
    plt.show()

log_losses = federated_df["log_loss"].tolist()
# Plot log loss
plt.figure(figsize=(8, 5))
plt.plot(federated_df["round"], log_losses, label="Federated Log Loss", marker='o')
plt.title("Log Loss Over Rounds")
plt.xlabel("Round")
plt.ylabel("Log Loss")
plt.grid(True)
plt.ylim(0, max(log_losses) * 1.1)
plt.legend()
plt.tight_layout()
plt.savefig("log_loss_comparison.png")
plt.show()

# show table of confusion matrix values over rounds
confusion_matrix = federated_df["confusion_matrix"].tolist()
confusion_matrix = [ast.literal_eval(x) for x in confusion_matrix]
# confusion_matrix has tp, tn, fp, fn
tp = [x["tp"] for x in confusion_matrix]
tn = [x["tn"] for x in confusion_matrix]
fp = [x["fp"] for x in confusion_matrix]
fn = [x["fn"] for x in confusion_matrix]
# Plot confusion matrix values
plt.figure(figsize=(8, 5))
plt.plot(federated_df["round"], tp, label="True Positives", marker='o')
plt.plot(federated_df["round"], tn, label="True Negatives", marker='o')
plt.plot(federated_df["round"], fp, label="False Positives", marker='o')
plt.plot(federated_df["round"], fn, label="False Negatives", marker='o')
plt.title("Confusion Matrix Values Over Rounds (Federated)")
plt.xlabel("Round")
plt.ylabel("Count")
plt.grid(True)
plt.ylim(0, max(max(tp), max(tn), max(fp), max(fn)) * 1.1)
plt.legend()
plt.tight_layout()
plt.savefig("confusion_matrix_comparison.png")
plt.show()

evaluation_time_ms = federated_df["evaluation_time_ms"].tolist()
# Plot evaluation time
plt.figure(figsize=(8, 5))
plt.plot(federated_df["round"], evaluation_time_ms, label="Federated Evaluation Time (ms)", marker='o')
plt.title("Evaluation Time Over Rounds")
plt.xlabel("Round")
plt.ylabel("Evaluation Time (ms)")
plt.grid(True)
plt.ylim(0, max(evaluation_time_ms) * 1.1)
plt.legend()
plt.tight_layout()
plt.savefig("evaluation_time_comparison.png")
plt.show()

prediction_confidence = federated_df["prediction_confidence"].tolist()
prediction_confidence = [ast.literal_eval(x) for x in prediction_confidence]
# prediction confidence has average, min, and max values
avg_confidence = [x["average"] for x in prediction_confidence]
min_confidence = [x["min"] for x in prediction_confidence]
max_confidence = [x["max"] for x in prediction_confidence]

# Plot prediction confidence
plt.figure(figsize=(8, 5))
plt.plot(federated_df["round"], avg_confidence, label="Average Prediction Confidence", marker='o')
plt.fill_between(federated_df["round"], min_confidence, max_confidence, color='gray', alpha=0.5, label="Confidence Range")
plt.title("Prediction Confidence Over Rounds (Federated)")
plt.xlabel("Round")
plt.ylabel("Prediction Confidence")
plt.grid(True)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig("prediction_confidence_comparison.png")
plt.show()

per_class_precision = federated_df["per_class_precision"].tolist()
per_class_precision = [ast.literal_eval(x) for x in per_class_precision]
# per_class_precision has precision for class_0 and class_1
class_0_precision = [x["class_0"] for x in per_class_precision]
class_1_precision = [x["class_1"] for x in per_class_precision]
# Plot per class precision
plt.figure(figsize=(8, 5))
plt.plot(federated_df["round"], class_0_precision, label="Class 0 Precision", marker='o')
plt.plot(federated_df["round"], class_1_precision, label="Class 1 Precision", marker='o')
plt.title("Per Class Precision Over Rounds (Federated)")
plt.xlabel("Round")
plt.ylabel("Precision")
plt.grid(True)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig("per_class_precision_comparison.png")
plt.show()

per_class_recall = federated_df["per_class_recall"].tolist()
per_class_recall = [ast.literal_eval(x) for x in per_class_recall]
# per_class_recall has recall for class_0 and class_1
class_0_recall = [x["class_0"] for x in per_class_recall]
class_1_recall = [x["class_1"] for x in per_class_recall]
# Plot per class recall
plt.figure(figsize=(8, 5))
plt.plot(federated_df["round"], class_0_recall, label="Class 0 Recall", marker='o')
plt.plot(federated_df["round"], class_1_recall, label="Class 1 Recall", marker='o')
plt.title("Per Class Recall Over Rounds")
plt.xlabel("Round")
plt.ylabel("Recall")
plt.grid(True)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig("per_class_recall_comparison.png")
plt.show()

with PdfPages("federated_vs_centralized_report.pdf") as pdf:
    for metric in metrics:
        plt.figure(figsize=(8, 5))
        for model_type in combined_df["model_type"].unique():
            data = combined_df[combined_df["model_type"] == model_type]
            plt.plot(data["round"], data[metric], label=model_type.capitalize(),
                     marker='o' if model_type == "federated" else 'd')
        plt.title(f"{metric.capitalize()} Over Rounds")
        plt.xlabel("Round")
        plt.ylabel(metric.capitalize())
        plt.grid(True)
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    
    # Log loss
    plt.figure(figsize=(8, 5))
    plt.plot(federated_df["round"], log_losses, label="Federated Log Loss", marker='o')
    plt.title("Log Loss Over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Log Loss")
    plt.grid(True)
    plt.ylim(0, max(log_losses) * 1.1)
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Confusion matrix values
    plt.figure(figsize=(8, 5))
    plt.plot(federated_df["round"], tp, label="True Positives", marker='o')
    plt.plot(federated_df["round"], tn, label="True Negatives", marker='o')
    plt.plot(federated_df["round"], fp, label="False Positives", marker='o')
    plt.plot(federated_df["round"], fn, label="False Negatives", marker='o')
    plt.title("Confusion Matrix Values Over Rounds (Federated)")
    plt.xlabel("Round")
    plt.ylabel("Count")
    plt.grid(True)
    plt.ylim(0, max(max(tp), max(tn), max(fp), max(fn)) * 1.1)
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Evaluation time
    plt.figure(figsize=(8, 5))
    plt.plot(federated_df["round"], evaluation_time_ms, label="Federated Evaluation Time (ms)", marker='o')
    plt.title("Evaluation Time Over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Evaluation Time (ms)")
    plt.grid(True)
    plt.ylim(0, max(evaluation_time_ms) * 1.1)
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    
    # Prediction confidence
    plt.figure(figsize=(8, 5))
    plt.plot(federated_df["round"], avg_confidence, label="Average Prediction Confidence", marker='o')
    plt.fill_between(federated_df["round"], min_confidence, max_confidence, color='gray', alpha=0.5,
                     label="Confidence Range")
    plt.title("Prediction Confidence Over Rounds (Federated)")
    plt.xlabel("Round")
    plt.ylabel("Prediction Confidence")
    plt.grid(True)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Per class precision
    plt.figure(figsize=(8, 5))
    plt.plot(federated_df["round"], class_0_precision, label="Class 0 Precision", marker='o')
    plt.plot(federated_df["round"], class_1_precision, label="Class 1 Precision", marker='o')
    plt.title("Per Class Precision Over Rounds (Federated)")
    plt.xlabel("Round")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Per class recall
    plt.figure(figsize=(8, 5))
    plt.plot(federated_df["round"], class_0_recall, label="Class 0 Recall", marker='o')
    plt.plot(federated_df["round"], class_1_recall, label="Class 1 Recall", marker='o')
    plt.title("Per Class Recall Over Rounds (Federated)")
    plt.xlabel("Round")
    plt.ylabel("Recall")
    plt.grid(True)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()


