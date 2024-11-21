import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from scipy import stats
from scipy.signal import welch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy, ks_2samp


def compute_zero_crossings(signal):
    """Compute zero crossing rate for a signal"""
    return np.sum(np.abs(np.diff(np.signbit(signal)))) / len(signal)


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


def comprehensive_statistical_analysis(file_paths, labels):
    """
    Perform comprehensive statistical analysis of ECG signals
    """
    results = {
        "temporal": {},
        "spectral": {},
        "statistical_tests": {},
        "effect_sizes": {},
        "zero_crossings": {"Normal": [], "AFIB": []},
    }

    # Load all signals
    normal_signals = []
    afib_signals = []

    for filepath, label in zip(file_paths, labels):
        signal = np.load(filepath).astype(np.float32)
        zcr = compute_zero_crossings(signal)

        if label == "N":
            normal_signals.append(signal)
            results["zero_crossings"]["Normal"].append(zcr)
        else:
            afib_signals.append(signal)
            results["zero_crossings"]["AFIB"].append(zcr)

    normal_signals = np.array(normal_signals)
    afib_signals = np.array(afib_signals)

    # 1. Basic Statistical Measures
    for name, signals in [("Normal", normal_signals), ("AFIB", afib_signals)]:
        results["temporal"][name] = {
            "mean": np.mean(signals, axis=1),
            "std": np.std(signals, axis=1),
            "skew": stats.skew(signals, axis=1),
            "kurtosis": stats.kurtosis(signals, axis=1),
        }

    # 2. Spectral Analysis
    def compute_psd(signal):
        freqs, psd = welch(signal, fs=300)  # Assuming 300Hz sampling rate
        return freqs, psd

    normal_psds = np.array([compute_psd(sig)[1] for sig in normal_signals])
    afib_psds = np.array([compute_psd(sig)[1] for sig in afib_signals])

    results["spectral"] = {
        "Normal": normal_psds,
        "AFIB": afib_psds,
        "freqs": compute_psd(normal_signals[0])[0],
    }

    # 3. Statistical Tests
    # KS test for each feature
    for feature in ["mean", "std", "skew", "kurtosis"]:
        ks_stat, p_value = ks_2samp(
            results["temporal"]["Normal"][feature], results["temporal"]["AFIB"][feature]
        )
        results["statistical_tests"][f"ks_test_{feature}"] = {
            "statistic": ks_stat,
            "p_value": p_value,
        }

    # Add zero crossing rate statistical test
    ks_stat, p_value = ks_2samp(
        results["zero_crossings"]["Normal"], results["zero_crossings"]["AFIB"]
    )
    results["statistical_tests"]["ks_test_zcr"] = {
        "statistic": ks_stat,
        "p_value": p_value,
    }

    # 4. Effect Sizes (Cohen's d)
    for feature in ["mean", "std", "skew", "kurtosis"]:
        n_mean = np.mean(results["temporal"]["Normal"][feature])
        a_mean = np.mean(results["temporal"]["AFIB"][feature])
        n_std = np.std(results["temporal"]["Normal"][feature])
        a_std = np.std(results["temporal"]["AFIB"][feature])

        pooled_std = np.sqrt((n_std**2 + a_std**2) / 2)
        cohen_d = (n_mean - a_mean) / (pooled_std + 1e-6)

        results["effect_sizes"][feature] = cohen_d

    # Add zero crossing rate effect size separately
    n_mean = np.mean(results["zero_crossings"]["Normal"])
    a_mean = np.mean(results["zero_crossings"]["AFIB"])
    n_std = np.std(results["zero_crossings"]["Normal"])
    a_std = np.std(results["zero_crossings"]["AFIB"])

    pooled_std = np.sqrt((n_std**2 + a_std**2) / 2)
    cohen_d = (n_mean - a_mean) / (pooled_std + 1e-6)
    results["effect_sizes"]["zcr"] = cohen_d

    return results

#
# def plot_analysis_results(results, length):
#     """
#     Create comprehensive visualization of analysis results and save individual subplots.
#     """
#     fig, axes = plt.subplots(4, 2, figsize=(15, 25))  # Changed from 3,2 to 4,2
#
#     # 1. Distribution plots for temporal features
#     features = ["mean", "std", "skew", "kurtosis"]
#     for i, feature in enumerate(features):
#         ax = axes[i // 2, i % 2]
#         normal_data = results["temporal"]["Normal"][feature]
#         afib_data = results["temporal"]["AFIB"][feature]
#
#         sns.kdeplot(normal_data, ax=ax, label="Normal")
#         sns.kdeplot(afib_data, ax=ax, label="AFIB")
#         ax.set_title(f"{feature} Distribution")
#         ax.legend()
#
#         # Add KS test results
#         ks_results = results["statistical_tests"][f"ks_test_{feature}"]
#         effect_size = results["effect_sizes"][feature]
#         ax.text(
#             0.05,
#             0.95,
#             f'KS p-value: {ks_results["p_value"]:.4f}\nCohen\'s d: {effect_size:.4f}',
#             transform=ax.transAxes,
#             verticalalignment="top",
#         )
#
#         # Save each plot as an individual PDF
#         plt.figure()
#         sns.kdeplot(normal_data, label="Normal")
#         sns.kdeplot(afib_data, label="AFIB")
#         plt.title(f"{feature} Distribution")
#         plt.legend()
#         plt.savefig(f"{feature}_distribution_{length}.pdf")
#         plt.close()
#
#     # 2. Average Power Spectral Density
#     ax = axes[2, 0]
#     freqs = results["spectral"]["freqs"]
#     normal_psd_mean = np.mean(results["spectral"]["Normal"], axis=0)
#     afib_psd_mean = np.mean(results["spectral"]["AFIB"], axis=0)
#
#     ax.plot(freqs, normal_psd_mean, label="Normal")
#     ax.plot(freqs, afib_psd_mean, label="AFIB")
#     ax.set_title("Average Power Spectral Density")
#     ax.set_xlabel("Frequency (Hz)")
#     ax.set_ylabel("Power")
#     ax.legend()
#
#     # Save PSD plot
#     plt.figure()
#     plt.plot(freqs, normal_psd_mean, label="Normal")
#     plt.plot(freqs, afib_psd_mean, label="AFIB")
#     plt.title("Average Power Spectral Density")
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Power")
#     plt.legend()
#     plt.savefig(f"average_psd_{length}.pdf")
#     plt.close()
#
#     # 3. Spectral KL Divergence
#     ax = axes[2, 1]
#     kl_divs = []
#     for n_psd, a_psd in zip(results["spectral"]["Normal"], results["spectral"]["AFIB"]):
#         # Normalize PSDs
#         n_psd_norm = n_psd / np.sum(n_psd)
#         a_psd_norm = a_psd / np.sum(a_psd)
#         kl_div = entropy(n_psd_norm, a_psd_norm)
#         kl_divs.append(kl_div)
#
#     sns.histplot(kl_divs, ax=ax)
#     ax.set_title("KL Divergence Distribution\nbetween Normal and AFIB spectra")
#
#     # Save KL divergence plot
#     plt.figure()
#     sns.histplot(kl_divs)
#     plt.title("KL Divergence Distribution\nbetween Normal and AFIB spectra")
#     plt.savefig(f"kl_divergence_{length}.pdf")
#     plt.close()
#
#     # Add Zero Crossing Rate plot separately
#     ax = axes[3, 0]
#     sns.kdeplot(results["zero_crossings"]["Normal"], ax=ax, label="Normal")
#     sns.kdeplot(results["zero_crossings"]["AFIB"], ax=ax, label="AFIB")
#     ax.set_title("Zero Crossing Rate Distribution")
#     ax.set_xlabel("Zero Crossing Rate")
#     ax.legend()
#
#     # Add KS test results and effect size
#     ks_results = results["statistical_tests"]["ks_test_zcr"]
#     effect_size = results["effect_sizes"]["zcr"]
#     ax.text(
#         0.05,
#         0.95,
#         f'KS p-value: {ks_results["p_value"]:.4f}\nCohen\'s d: {effect_size:.4f}',
#         transform=ax.transAxes,
#         verticalalignment="top",
#     )
#
#     # Save Zero Crossing Rate plot
#     plt.figure()
#     sns.kdeplot(results["zero_crossings"]["Normal"], label="Normal")
#     sns.kdeplot(results["zero_crossings"]["AFIB"], label="AFIB")
#     plt.title("Zero Crossing Rate Distribution")
#     plt.xlabel("Zero Crossing Rate")
#     plt.legend()
#     plt.savefig(f"zero_crossing_rate_{length}.pdf")
#     plt.close()
#
#     axes[3, 1].remove()  # Remove the unused subplot
#
#     plt.tight_layout()
#     return fig

def plot_analysis_results(results, length):
    """
    Create comprehensive visualization of analysis results and save individual subplots.
    """
    fig, axes = plt.subplots(4, 2, figsize=(15, 25))  # Changed from 3,2 to 4,2

    # 1. Distribution plots for temporal features
    features = ["mean", "std", "skew", "kurtosis"]
    for i, feature in enumerate(features):
        ax = axes[i // 2, i % 2]
        normal_data = results["temporal"]["Normal"][feature]
        afib_data = results["temporal"]["AFIB"][feature]

        sns.kdeplot(normal_data, ax=ax, label="Normal")
        sns.kdeplot(afib_data, ax=ax, label="AFIB")
        ax.set_title(f"{feature} Distribution")
        ax.legend()

        # Add KS test results
        ks_results = results["statistical_tests"][f"ks_test_{feature}"]
        effect_size = results["effect_sizes"][feature]
        ax.text(
            0.05,
            0.95,
            f'KS p-value: {ks_results["p_value"]:.4f}\nCohen\'s d: {effect_size:.4f}',
            transform=ax.transAxes,
            verticalalignment="top",
        )

        # Save each plot as an individual PDF
        plt.figure()
        sns.kdeplot(normal_data, label="Normal")
        sns.kdeplot(afib_data, label="AFIB")
        plt.title(f"{feature} Distribution")
        plt.legend()
        plt.savefig(f"{feature}_distribution_{length}.pdf")
        plt.close()

    # 2. Average Power Spectral Density
    ax = axes[2, 0]
    freqs = results["spectral"]["freqs"]
    normal_psd_mean = np.mean(results["spectral"]["Normal"], axis=0)
    afib_psd_mean = np.mean(results["spectral"]["AFIB"], axis=0)

    ax.plot(freqs, normal_psd_mean, label="Normal")
    ax.plot(freqs, afib_psd_mean, label="AFIB")
    ax.set_title("Average Power Spectral Density")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power ()")
    ax.legend()

    # Save PSD plot
    plt.figure()
    plt.plot(freqs, normal_psd_mean, label="Normal")
    plt.plot(freqs, afib_psd_mean, label="AFIB")
    plt.title("Average Power Spectral Density")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (mV\u00b2/Hz)")
    plt.legend()
    plt.savefig(f"average_psd_{length}.pdf")
    plt.close()

    # Add normalized spectrum with vertical lines
    plt.figure()
    plt.vlines(freqs, 0, normal_psd_mean / np.sum(normal_psd_mean), colors='blue', label="Normal (Normalized)", alpha=0.6)
    plt.vlines(freqs, 0, afib_psd_mean / np.sum(afib_psd_mean), colors='orange', label="AFIB (Normalized)", alpha=0.6)
    plt.title("Normalized Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Power")
    plt.legend("PSD (mV\u00b2/Hz)")
    plt.savefig(f"normalized_spectrum_{length}.pdf")
    plt.close()

    # 3. Spectral KL Divergence
    ax = axes[2, 1]
    kl_divs = []
    for n_psd, a_psd in zip(results["spectral"]["Normal"], results["spectral"]["AFIB"]):
        # Normalize PSDs
        n_psd_norm = n_psd / np.sum(n_psd)
        a_psd_norm = a_psd / np.sum(a_psd)
        kl_div = entropy(n_psd_norm, a_psd_norm)
        kl_divs.append(kl_div)

    sns.histplot(kl_divs, ax=ax)
    ax.set_title("KL Divergence Distribution\nbetween Normal and AFIB spectra")

    # Save KL divergence plot
    plt.figure()
    sns.histplot(kl_divs)
    plt.title("KL Divergence Distribution\nbetween Normal and AFIB spectra")
    plt.savefig(f"kl_divergence_{length}.pdf")
    plt.close()

    # Add Zero Crossing Rate plot separately
    ax = axes[3, 0]
    sns.kdeplot(results["zero_crossings"]["Normal"], ax=ax, label="Normal")
    sns.kdeplot(results["zero_crossings"]["AFIB"], ax=ax, label="AFIB")
    ax.set_title("Zero Crossing Rate Distribution")
    ax.set_xlabel("Zero Crossing Rate")
    ax.legend()

    # Add KS test results and effect size
    ks_results = results["statistical_tests"]["ks_test_zcr"]
    effect_size = results["effect_sizes"]["zcr"]
    ax.text(
        0.05,
        0.95,
        f'KS p-value: {ks_results["p_value"]:.4f}\nCohen\'s d: {effect_size:.4f}',
        transform=ax.transAxes,
        verticalalignment="top",
    )

    # Save Zero Crossing Rate plot
    plt.figure()
    sns.kdeplot(results["zero_crossings"]["Normal"], label="Normal")
    sns.kdeplot(results["zero_crossings"]["AFIB"], label="AFIB")
    plt.title("Zero Crossing Rate Distribution")
    plt.xlabel("Zero Crossing Rate")
    plt.legend()
    plt.savefig(f"zero_crossing_rate_{length}.pdf")
    plt.close()

    axes[3, 1].remove()  # Remove the unused subplot

    plt.tight_layout()
    return fig



def start_analysis():
    """
    Perform comprehensive statistical analysis on the dataset.
    """
    length = "10k"
    # Prepare data
    csv_path = f"{length}\\REFERENCE_test1030.csv"
    base_dir = f"{length}"
    # Load your data
    data_splits = prepare_data_splits(csv_path, base_dir)

    # Combine all data for analysis
    all_files = data_splits["train"][0] + data_splits["val"][0] + data_splits["test"][0]
    all_labels = (
        data_splits["train"][1] + data_splits["val"][1] + data_splits["test"][1]
    )

    # Perform analysis
    results = comprehensive_statistical_analysis(all_files, all_labels)

    # Create visualization
    fig = plot_analysis_results(results, length)
    plt.savefig(f"statistical_analysis_{length}.png")

    # Print summary statistics
    print("\nStatistical Analysis Summary:")
    print("\nKolmogorov-Smirnov Tests:")
    for feature, test_results in results["statistical_tests"].items():
        print(f"{feature}:")
        print(f"  p-value: {test_results['p_value']:.4f}")

    print("\nEffect Sizes (Cohen's d):")
    for feature, effect_size in results["effect_sizes"].items():
        print(f"{feature}: {effect_size:.4f}")


if __name__ == "__main__":
    start_analysis()