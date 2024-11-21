import os
import torch
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
# from src.binary_classification.data_simple1014 import load_ekg_data
from src.binary_classification.data_1lead import load_ekg_data
from src.utils.torchutils import set_seed, load_model
from src.utils.metrics import BinaryAccuracy

# static variables
NUM_CLASSES: int = 2

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)

def plot_confusion_matrix(y_true, y_pred, class_names, accuracy, f1, dataset_name):
    """
    Plots a confusion matrix using sklearn's ConfusionMatrixDisplay.
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: List of class names.
        accuracy: Accuracy of the model.
        f1: F1 score of the model.
        dataset_name: Name of the dataset.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    # Add accuracy and F1 text to the plot
    plt.title(f"Confusion Matrix - {dataset_name}\nAccuracy: {accuracy:.4f} | F1 Score: {f1:.4f}")
    plt.savefig(f"{dataset_name}_{accuracy}_CM.png")
    plt.show()

def evaluate_model_on_dataset(name, data_path, dataset_name, accuracy: BinaryAccuracy = BinaryAccuracy()):
    """
    This function evaluates the model on a specific dataset, generates the confusion matrix,
    and measures inference time and throughput.
    """
    # Load test data
    _, _, test_data = load_ekg_data(data_path, num_workers=4, num_classes=NUM_CLASSES)

    # Load model
    model = load_model(name).to(device).float()
    model.eval()

    all_preds = []
    all_labels = []

    total_inference_time = 0.0  # To accumulate total inference time
    total_samples = 0           # To accumulate total samples processed

    with torch.no_grad():
        for inputs, targets in test_data:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.permute(0, 2, 1)

            # Start time for inference
            start_time = time.time()

            # Forward pass
            outputs = model(inputs)

            # End time for inference
            batch_inference_time = time.time() - start_time
            total_inference_time += batch_inference_time

            # Count total samples in this batch
            total_samples += inputs.size(0)

            # Convert model output to predicted class (assuming outputs are logits)
            preds = torch.argmax(outputs, dim=1)

            # Adjust targets based on the one-hot encoding order
            labels = torch.argmax(targets, dim=1)  # Convert one-hot targets to class labels

            # Collect predictions and labels for confusion matrix and F1
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Compute accuracy
            accuracy.update(preds, labels)

    # Show accuracy
    accuracy_value = accuracy.compute()
    print(f"Accuracy on {dataset_name}: {accuracy_value:.4f}")
    accuracy.reset()

    # Calculate F1 score
    f1_value = f1_score(all_labels, all_preds, average='weighted')
    print(f"F1 Score on {dataset_name}: {f1_value:.4f}")

    # Plot confusion matrix
    class_names = ["AFIB", "Normal"]
    plot_confusion_matrix(all_labels, all_preds, class_names, round(accuracy_value, 4), round(f1_value, 4), dataset_name)

    # Calculate throughput (samples per second)
    throughput = total_samples / total_inference_time if total_inference_time > 0 else 0.0

    print(f"Total inference time: {total_inference_time:.4f} seconds")
    print(f"Inference throughput: {throughput:.2f} records/second")


if __name__ == "__main__":
    length = '1k'
    name = 'Mamca'
    # model_path = r"C:\wenjian\MasterArbeit\Code\repo\My_Mamba_ECG_Classification\Lichtenberg\results\final\Mamba\1k\best_model_20241021_174157_binary_MambaBEAT_1k_0.7987.pth"
    # model_path = rf"C:\wenjian\MasterArbeit\Code\repo\test_repo\models\best_model_20241116_225227_binary_MAMCA_1k_0.7217.pth"

    # TRAINING_DATA_PATH: str = rf"C:\wenjian\MasterArbeit\Code\dataset\Icential11k_dataset\1k"
    current_file_path = os.path.abspath(__file__)
    current_folder = os.path.dirname(current_file_path)
    TRAINING_DATA_PATH = os.path.abspath(os.path.join(current_folder, "..", "..", "..", "dataset/Icential11k_dataset/1k"))
    model_path = os.path.abspath(os.path.join(current_folder, "..", "models", "best_model_20241121_214402_binary_MAMCA_1k_0.8793.pth"))
    evaluate_model_on_dataset(model_path, TRAINING_DATA_PATH, f"{name}_{length}", BinaryAccuracy())
