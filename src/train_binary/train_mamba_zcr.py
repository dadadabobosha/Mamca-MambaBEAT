import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import math
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
import torch.nn.functional as F


class SelectiveScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A_in, X_in):
        # Store the original sequence length
        original_L = A_in.size(1)
        last_elementA = A_in.transpose(2, 1)[:, :, -1:]
        last_elementX = X_in.transpose(2, 1)[:, :, -1:]

        # Calculate the next power of 2
        next_power_of_2 = 2 ** math.ceil(math.log2(original_L))

        # Extend the sequence lengths to the next power of 2 with zeros
        A = F.pad(A_in, (0, 0, 0, 0, 0, next_power_of_2 - original_L), "constant", 0)
        X = F.pad(X_in, (0, 0, 0, 0, 0, next_power_of_2 - original_L), "constant", 0)
        L = A.size(1)

        # Transpose the input tensors for efficient memory access during computation
        A = A.transpose(2, 1)
        X = X.transpose(2, 1)

        # Calculate the number of iterations needed
        iterations = int(math.log2(L))

        # Perform the up-sweep operation
        for d in range(iterations):
            indices = torch.arange(0, L, 2 ** (d + 1))
            X[:, :, indices + 2 ** (d + 1) - 1] += (
                A[:, :, indices + 2 ** (d + 1) - 1] * X[:, :, indices + 2**d - 1]
            )
            A[:, :, indices + 2 ** (d + 1) - 1] *= A[:, :, indices + 2**d - 1]

        # Perform the down-sweep operation
        X[:, :, -1] = 0
        for d in range(iterations - 1, -1, -1):
            indices = torch.arange(0, X.size(2), 2 ** (d + 1))
            t = X[:, :, indices + 2**d - 1].clone()
            X[:, :, indices + 2**d - 1] = X[:, :, indices + 2 ** (d + 1) - 1]
            X[:, :, indices + 2 ** (d + 1) - 1] = (
                A[:, :, indices + 2**d - 1] * X[:, :, indices + 2 ** (d + 1) - 1] + t
            )

        # Remove the first zero elements and add the last elements
        X = torch.cat(
            (
                X[:, :, 1:original_L],
                last_elementA * X[:, :, original_L - 1 : original_L] + last_elementX,
            ),
            dim=2,
        )
        X = X.transpose(2, 1)

        # Save tensors for backward pass
        ctx.save_for_backward(A_in, X)
        return X

    @staticmethod
    def backward(ctx, grad_output):
        A_in, X = ctx.saved_tensors

        # Store the original sequence length
        original_L = grad_output.size(1)

        # Calculate the next power of 2
        next_power_of_2 = 2 ** math.ceil(math.log2(original_L))

        # Extend the sequence lengths to the next power of 2 with zeros
        grad_output = F.pad(
            grad_output, (0, 0, 0, 0, 0, next_power_of_2 - original_L), "constant", 0
        )
        A = F.pad(A_in, (0, 0, 0, 0, 0, next_power_of_2 - original_L), "constant", 0)
        X = F.pad(X, (0, 0, 0, 0, 0, next_power_of_2 - original_L), "constant", 0)

        # Transpose the tensors for efficient memory access during computation
        grad_output = grad_output.transpose(2, 1)
        A = A.transpose(2, 1)
        X = X.transpose(2, 1)

        # Shift A one to the left
        A = F.pad(A[:, :, 1:], (0, 0, 0, 1))
        B, D, L, _ = A.size()

        # Calculate the number of iterations needed
        iterations = int(math.log2(L))

        # Perform the up-sweep operation
        for d in range(iterations):
            indices = torch.arange(0, L, 2 ** (d + 1))
            grad_output[:, :, indices] += (
                A[:, :, indices] * grad_output[:, :, indices + 2**d]
            )
            A[:, :, indices] *= A[:, :, indices + 2**d]

        # Perform the down-sweep operation
        Aa = A
        Xa = grad_output

        for d in range(iterations - 1, -1, -1):
            Aa = A[:, :, 0 : L : 2**d]
            Xa = grad_output[:, :, 0 : L : 2**d]

            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, :-1, 1].add_(Aa[:, :, :-1, 1].mul(Xa[:, :, 1:, 0]))
            Aa[:, :, :-1, 1].mul_(Aa[:, :, 1:, 0])

        # # Perform the down-sweep operation
        # grad_output[:, :, -1] = 0
        # for d in range(iterations - 1, -1, -1):
        #     indices = torch.arange(0, L, 2 ** (d + 1))
        #     t = grad_output[:, :, indices + 2 ** d].clone()
        #     grad_output[:, :, indices + 2 ** d] = grad_output[:, :, indices]
        #     grad_output[:, :, indices] = (
        #         A[:, :, indices] * grad_output[:, :, indices + 2 ** d] + t
        #     )

        # Compute gradient with respect to A
        grad_A = torch.zeros_like(X)
        grad_A[:, :, 1:] = X[:, :, :-1] * grad_output[:, :, 1:]

        # Return back to original dimensions
        return (
            grad_A.transpose(2, 1)[:, :original_L],
            grad_output.transpose(2, 1)[:, :original_L],
        )


def selective_scan(A_in, X_in):
    return SelectiveScan.apply(A_in, X_in)


class MambaBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_state_dim: int,
        expand: int,
        dt_rank: int,
        kernel_size: int,
        conv_bias: bool,
        bias: bool,
        method: str,
    ):
        super(MambaBlock, self).__init__()
        self.in_channels = in_channels
        self.latent_state_dim = latent_state_dim
        self.expand = expand
        self.dt_rank = dt_rank
        self.kernel_size = kernel_size
        self.method = method

        self.expanded_dim = int(self.expand * self.in_channels)
        self.in_proj = torch.nn.Linear(
            self.in_channels, self.expanded_dim * 2, bias=bias
        )

        self.conv1d = torch.nn.Conv1d(
            in_channels=self.expanded_dim,
            out_channels=self.expanded_dim,
            bias=conv_bias,
            kernel_size=kernel_size,
            groups=self.expanded_dim,
            padding=kernel_size - 1,
        )

        self.activation = torch.nn.SiLU()

        self.selection = torch.nn.Linear(
            self.expanded_dim, self.latent_state_dim * 2 + self.dt_rank, bias=False
        )
        self.dt_proj = torch.nn.Linear(
            self.dt_rank, self.expanded_dim, bias=True
        )  # Broadcast

        # S4D Initialization
        A = torch.arange(1, self.latent_state_dim + 1, dtype=torch.float32).repeat(
            self.expanded_dim, 1
        )
        self.A_log = torch.nn.Parameter(torch.log(A))
        self.D = torch.nn.Parameter(torch.ones(self.expanded_dim, dtype=torch.float32))

        self.out_proj = torch.nn.Linear(self.expanded_dim, self.in_channels, bias=bias)

    def forward(self, x):
        L = x.size(1)
        # Project input x an residual connection z
        x_z = self.in_proj(x)

        # Split expanded x and residual z
        x, z = x_z.chunk(2, dim=-1)

        # pass input through the conv and the non_linearity
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)

        x = self.activation(x)

        # Compute ssm -> ssm(Ad, Bd, C, D) or ssm(A, B, C, D, dt)
        out = self.selective_ssm(x)

        # Activation of residual connection:
        z = self.activation(z)

        # multiply outputs by residual connection
        out *= z

        # and calculate output
        out = self.out_proj(out)

        return out

    def selective_ssm(self, x):
        A = -torch.exp(self.A_log)

        # Get B, C and dt from self.selection
        B_C_dt = self.selection(x)

        # Split the matrix.
        B, C, dt = torch.split(
            B_C_dt, [self.latent_state_dim, self.latent_state_dim, self.dt_rank], dim=-1
        )

        # Broadcast dt with self.dt_proj
        dt = torch.nn.functional.softplus(self.dt_proj(dt))
        Ad, Bd = self.discretize(dt, A, B, self.method)
        hidden = selective_scan(Ad, Bd * x.unsqueeze(-1))

        out = hidden @ C.unsqueeze(-1)

        return out.squeeze(3) + self.D * x

    @staticmethod
    def discretize(dt, A, B, method):
        if method == "zoh":
            # Zero-Order Hold (ZOH) method
            Ad = torch.exp(dt.unsqueeze(-1) * A)
            Bd = dt.unsqueeze(-1) * B.unsqueeze(2)
        elif method == "bilinear":
            raise NotImplementedError
            # TODO: complete the method
            # E = torch.eye(A.size(0), dtype=A.dtype, device=A.device)
            # half_dt_A = 0.5 * dt.unsqueeze(-1) * A
            # Ad = torch.inverse(E - half_dt_A) @ (E + half_dt_A)
            # Bd = torch.inverse(E - half_dt_A) @ dt * B
        else:
            raise ValueError("Invalid method. Choose either 'zoh' or 'bilinear'.")

        return Ad, Bd


class MambaBEAT(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_layers: int = 1,
        latent_state_dim: int = 16,
        expand: int = 2,
        dt_rank: int = None,
        kernel_size: int = 4,
        conv_bias: bool = True,
        bias: bool = True,
        method: str = "zoh",
        dropout: float = 0,
    ):
        super().__init__()

        if dt_rank is None:
            dt_rank = math.ceil(in_channels / latent_state_dim)

        self.layers = torch.nn.Sequential(
            *[
                MambaBlock(
                    in_channels,
                    latent_state_dim,
                    expand,
                    dt_rank,
                    kernel_size,
                    conv_bias,
                    bias,
                    method,
                )
                for _ in range(n_layers)
            ]
        )

        self.norm = RMSNorm(in_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # Get last batch of labels
        x = self.layers(x)[:, -1]
        x = self.norm(x)
        x = self.dropout(x)
        x = self.linear(x)
        return self.softmax(x)  # Returns probabilities for both classes


class RMSNorm(torch.nn.Module):
    def __init__(self, size: int, epsilon: float = 1e-5, bias: bool = False):
        super().__init__()

        self.epsilon = epsilon
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.bias = torch.nn.Parameter(torch.zeros(size)) if bias else None

    def forward(self, x):
        normed_x = (
            x
            * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.epsilon)
            * self.weight
        )

        if self.bias is not None:
            return normed_x + self.bias

        return normed_x


def prepare_data_splits(csv_path, base_dir):
    """
    Prepare train/val/test splits based on patient IDs
    """
    # Load the reference CSV
    df = pd.read_csv(csv_path, header=None)
    df.columns = ["file_name", "label"]

    # Extract patient IDs from file names
    df["patient_id"] = df["file_name"].apply(lambda x: x.split("_")[0])

    # Get unique patients and their labels
    unique_patients = df["patient_id"].unique()
    patient_labels = [
        df[df["patient_id"] == pid]["label"].iloc[0] for pid in unique_patients
    ]

    # First split: separate train from test/val
    train_patients, temp_patients = train_test_split(
        unique_patients, test_size=0.7, random_state=32, stratify=patient_labels
    )

    # Second split: separate test and val
    temp_patient_labels = [
        df[df["patient_id"] == pid]["label"].iloc[0] for pid in temp_patients
    ]
    val_patients, test_patients = train_test_split(
        temp_patients, test_size=0.5, random_state=32, stratify=temp_patient_labels
    )

    # Create dataframes for each split
    train_df = df[df["patient_id"].isin(train_patients)]
    val_df = df[df["patient_id"].isin(val_patients)]
    test_df = df[df["patient_id"].isin(test_patients)]

    # Create file paths
    def create_file_paths(split_df):
        return [
            os.path.join(base_dir, row["label"], row["file_name"])
            for _, row in split_df.iterrows()
        ]

    train_paths = create_file_paths(train_df)
    val_paths = create_file_paths(val_df)
    test_paths = create_file_paths(test_df)

    # Get labels
    train_labels = train_df["label"].tolist()
    val_labels = val_df["label"].tolist()
    test_labels = test_df["label"].tolist()

    # Calculate class weights for training
    label_counts = pd.Series(train_labels).value_counts()
    class_weights = 1.0 / label_counts
    sample_weights = [class_weights[label] for label in train_labels]

    return {
        "train": (train_paths, train_labels, sample_weights),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels),
    }


def compute_zero_crossings(signal):
    """Compute zero crossing rate"""
    return np.sum(np.abs(np.diff(np.signbit(signal)))) / len(signal)


class ECGDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = [1 if label == "AFIB" else 0 for label in labels]

        # Load signals and compute ZCR
        signals, zcrs = self._load_data()

        # Standardize ZCR
        scaler = StandardScaler()
        self.zcrs = scaler.fit_transform(zcrs.reshape(-1, 1))
        self.signals = torch.FloatTensor(signals)

    def _load_data(self):
        signals = []
        zcrs = []

        for filepath in tqdm(self.file_paths, desc="Loading signals"):
            signal = np.load(filepath).astype(np.float32)

            # Compute ZCR
            zcr = compute_zero_crossings(signal)
            zcrs.append(zcr)

            # Normalize signal
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
            signals.append(signal)

        return np.array(signals), np.array(zcrs)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return (
            self.signals[idx],
            torch.FloatTensor(self.zcrs[idx]).view(1),  # Shape: (1,)
            torch.tensor(self.labels[idx], dtype=torch.long),  # Changed to long
        )


class MambaZCR(torch.nn.Module):
    """Memory-optimized MambaBEAT with ZCR attention"""

    def __init__(self, sequence_length=1000, hidden_dim=64, out_channels=2):
        super().__init__()

        # Reduced complexity signal preprocessing
        self.signal_preprocessing = torch.nn.Sequential(
            torch.nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16, hidden_dim, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
        )

        # Memory-efficient Mamba configuration
        self.mamba = MambaBEAT(
            in_channels=hidden_dim,
            out_channels=32,
            n_layers=1,
            latent_state_dim=16,
            expand=2,
            dt_rank=8,
            kernel_size=4,
            dropout=0.1,
        )

        # Lightweight ZCR processing
        self.zcr_processing = torch.nn.Sequential(
            torch.nn.Linear(1, 16),
            torch.nn.LayerNorm(16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.LayerNorm(32),
            torch.nn.ReLU(),
        )

        # Simple feature combination
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(64, 32),  # 32 from Mamba + 32 from ZCR
            torch.nn.LayerNorm(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(32, out_channels),
        )

    def forward(self, signal, zcr):
        batch_size = signal.size(0)

        # Process signal with memory efficiency
        signal = signal.unsqueeze(1)  # Add channel dimension

        # Downsample and process signal
        signal_features = self.signal_preprocessing(signal)
        signal_features = signal_features.transpose(1, 2)

        # Process through Mamba
        mamba_features = self.mamba(signal_features)  # Shape: (batch_size, 32)

        # Process ZCR and ensure correct shape
        zcr_features = self.zcr_processing(zcr)  # Shape: (batch_size, 1, 32)
        zcr_features = zcr_features.squeeze(1)  # Remove the extra dimension

        # Simple concatenation (both should be 2D now)
        combined_features = torch.cat([mamba_features, zcr_features], dim=1)

        # Classification
        logits = self.classifier(combined_features)
        return torch.nn.functional.softmax(logits, dim=1)


def train_model(model, train_loader, val_loader, num_epochs, device):
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.00)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True
    )

    best_val_auc = 0
    patience = 30
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for signals, zcrs, labels in pbar:
            # Clear cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            signals = signals.float().to(device)
            zcrs = zcrs.float().to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad()
            outputs = model(signals, zcrs)
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            train_loss += loss.item()
            all_preds.extend(outputs[:, 1].detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Clear unnecessary tensors
            del outputs, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            pbar.set_postfix({"loss": train_loss / len(train_loader)})

        # Validation phase
        val_metrics = evaluate_model(model, val_loader, device)
        print(f"\nEpoch {epoch+1} - Validation Metrics:")
        for name, value in val_metrics.items():
            if name != "confusion_matrix":
                print(f"{name}: {value:.4f}")

        scheduler.step(val_metrics["auc"])

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            torch.save(model.state_dict(), "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break


def evaluate_model(model, data_loader, device):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for signals, zcrs, labels in data_loader:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            signals = signals.float().to(device)
            zcrs = zcrs.float().to(device)

            outputs = model(signals, zcrs)
            predictions.extend(outputs[:, 1].cpu().numpy())
            true_labels.extend(labels.numpy())

            del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    binary_preds = (predictions > 0.5).astype(int)

    return {
        "accuracy": accuracy_score(true_labels, binary_preds),
        "auc": roc_auc_score(true_labels, predictions),
        "f1": f1_score(true_labels, binary_preds),
        "precision": precision_score(true_labels, binary_preds),
        "recall": recall_score(true_labels, binary_preds),
        "confusion_matrix": confusion_matrix(true_labels, binary_preds),
    }


def main():
    batch_size = 64
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = r"C:\wenjian\MasterArbeit\Code\dataset\Icential11k_dataset\1k"
    csv_path = rf"{base_dir}\REFERENCE.csv"

    # Prepare data
    data_splits = prepare_data_splits(csv_path, base_dir)

    # Create datasets and loaders
    train_dataset = ECGDataset(data_splits["train"][0], data_splits["train"][1])
    val_dataset = ECGDataset(data_splits["val"][0], data_splits["val"][1])
    test_dataset = ECGDataset(data_splits["test"][0], data_splits["test"][1])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize and train model
    model = MambaZCR(sequence_length=1000, hidden_dim=1).to(device)
    train_model(model, train_loader, val_loader, num_epochs, device)

    # Load saved model for testing
    model.load_state_dict(torch.load("best_model.pth"))

    # Test evaluation
    test_metrics = evaluate_model(model, test_loader, device)

    print("\nTest Set Metrics:")
    for name, value in test_metrics.items():
        if name != "confusion_matrix":
            print(f"{name}: {value:.4f}")
    print("\nConfusion Matrix:")
    print(test_metrics["confusion_matrix"])


if __name__ == "__main__":
    main()
