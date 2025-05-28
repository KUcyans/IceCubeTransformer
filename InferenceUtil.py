import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from typing import Tuple
import sys
import pandas as pd
import ast
from Enum.Flavour import Flavour
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
# matplotlib.use("Agg")

sys.path.append('/groups/icecube/cyan/Utils')
from PlotUtils import setMplParam, getColour, getHistoParam 
# getHistoParam:
# Nbins, binwidth, bins, counts, bin_centers  = 
from DB_lister import list_content, list_tables
from ExternalFunctions import nice_string_output, add_text_to_ax
setMplParam(isVaryLineStyle=False)

def get_nu_prob(df: pd.DataFrame, flavour: Flavour) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """it returns the probability of each flavour in the dataframe.
    nu_e_prob is the probability of target(flavour) of true nu_e
    nu_mu_prob is the probability of target(flavour) of true nu_mu
    nu_tau_prob is the probability of target(flavour) of true nu_tau  

    Args:
        df (pd.DataFrame): prediction csv
        flavour (Flavour): target flavour

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: _description_
    """
    def safe_parse(x):
        return ast.literal_eval(x) if isinstance(x, str) else x

    df["prob"] = df["prob"].apply(safe_parse)

    nu_e = df[df["target_class"] == 0]
    nu_mu = df[df["target_class"] == 1]
    nu_tau = df[df["target_class"] == 2]

    index = 0 if flavour == Flavour.E else 1 if flavour == Flavour.MU else 2 if flavour == Flavour.TAU else None

    if index is None:
        raise ValueError(f"Unknown flavour: {flavour}")

    nu_e_prob = nu_e["prob"].apply(lambda x: x[index]).to_numpy()
    nu_mu_prob = nu_mu["prob"].apply(lambda x: x[index]).to_numpy()
    nu_tau_prob = nu_tau["prob"].apply(lambda x: x[index]).to_numpy()

    return nu_e_prob, nu_mu_prob, nu_tau_prob

def get_energy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    def safe_parse(x):
        return ast.literal_eval(x) if isinstance(x, str) else x

    df["energy"] = df["energy"].apply(safe_parse)

    nu_e = df[df["target_class"] == 0]
    nu_mu = df[df["target_class"] == 1]
    nu_tau = df[df["target_class"] == 2]

    nu_e_energy = nu_e["energy"].to_numpy()
    nu_mu_energy = nu_mu["energy"].to_numpy()
    nu_tau_energy = nu_tau["energy"].to_numpy()

    return nu_e_energy, nu_mu_energy, nu_tau_energy

def get_zenith(df:pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    def safe_parse(x):
        return ast.literal_eval(x) if isinstance(x, str) else x

    df["zenith"] = df["zenith"].apply(safe_parse) # in radian

    nu_e = df[df["target_class"] == 0]
    nu_mu = df[df["target_class"] == 1]
    nu_tau = df[df["target_class"] == 2]

    nu_e_zenith = nu_e["zenith"].to_numpy()
    nu_mu_zenith = nu_mu["zenith"].to_numpy()
    nu_tau_zenith = nu_tau["zenith"].to_numpy()
    nu_e_zenith = np.rad2deg(nu_e_zenith)
    nu_mu_zenith = np.rad2deg(nu_mu_zenith)
    nu_tau_zenith = np.rad2deg(nu_tau_zenith)
    return nu_e_zenith, nu_mu_zenith, nu_tau_zenith


## Plotting Functions
def plot_binary_flavour_ROC(df: pd.DataFrame, signal_flavour: Flavour, id: str) -> None:
    fig, ax = plt.subplots(figsize=(17, 11))
    nu_e_prob, nu_mu_prob, nu_tau_prob = get_nu_prob(df, signal_flavour)

    all_outputs = {
        Flavour.E: nu_e_prob,
        Flavour.MU: nu_mu_prob,
        Flavour.TAU: nu_tau_prob
    }

    y_true = []
    y_score = []

    for flavour, output in all_outputs.items():
        y_true.append(np.ones_like(output) if flavour == signal_flavour else np.zeros_like(output))
        y_score.append(output)

    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # make a residual flavour name list, by excluding the signal flavour from all_logits.keys()
    residual_flavours = [flavour for flavour in all_outputs.keys() if flavour != signal_flavour]
    label_roc = (
        fr"$\bf{signal_flavour.latex}$ vs "
        fr"(${residual_flavours[0].latex}$+${residual_flavours[1].latex}$)"
        + "\n" + f"AUC = {roc_auc:.3f}"
    )
    set_colour = getColour(2) if signal_flavour == Flavour.E else getColour(0) if signal_flavour == Flavour.MU else getColour(1)
    ax.plot(fpr, tpr, linewidth=2, color=set_colour, markersize=1, label=label_roc)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=24)
    ax.set_ylabel('True Positive Rate', fontsize=24)
    ax.set_title(fr"{id}: ${signal_flavour.latex}$ vs (${residual_flavours[0].latex}$+${residual_flavours[1].latex}$)")
    ax.legend(loc="lower right", fontsize=24)
    plt.tight_layout()
    plt.show()

def plot_multi_flavour_ROC(df: pd.DataFrame, id: str) -> None:
    fig, ax = plt.subplots(figsize=(17, 11))

    ref_fpr = None
    ref_thresholds = None

    for colour_i, flavour in [(2, Flavour.E), (0, Flavour.MU), (1, Flavour.TAU)]:
        nu_e_prob, nu_mu_prob, nu_tau_prob = get_nu_prob(df, flavour)

        y_true = np.concatenate([
            np.ones_like(nu_e_prob) if flavour == Flavour.E else np.zeros_like(nu_e_prob),
            np.ones_like(nu_mu_prob) if flavour == Flavour.MU else np.zeros_like(nu_mu_prob),
            np.ones_like(nu_tau_prob) if flavour == Flavour.TAU else np.zeros_like(nu_tau_prob),
        ])
        y_score = np.concatenate([nu_e_prob, nu_mu_prob, nu_tau_prob])

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, label=fr"${flavour.latex}$ (AUC = {roc_auc:.3f})",
                linewidth=2, color=getColour(colour_i), markersize=1)

        if ref_fpr is None and ref_thresholds is None:
            ref_fpr = fpr
            ref_thresholds = thresholds

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=24)
    ax.set_ylabel('True Positive Rate', fontsize=24)
    ax.set_title(rf"{id}: $\nu_{{\alpha}}$ vs ($\nu_{{\beta}}$+$\nu_{{\gamma}}$)")
    ax.legend(loc="lower right", fontsize=24)
    plt.tight_layout()
    plt.show()

def fraction_above_threshold(arr: np.ndarray, threshold: float) -> float:
    return np.mean(arr > threshold)

def fraction_below_threshold(arr: np.ndarray, threshold: float) -> float:
    return np.mean(arr < threshold)

def plot_prob_distribution(df: pd.DataFrame,
                            flavour: Flavour,
                            id: str,
                            manifier: Tuple[float, float] = None) -> None:
    # nu_e_logit, nu_mu_logit, nu_tau_logit = get_nu_logits(df, flavour)
    
    nu_e_prob, nu_mu_prob, nu_tau_prob = get_nu_prob(df, flavour)
    Nbins, binwidth, bins, counts, bin_centers = getHistoParam(nu_tau_prob, binwidth=0.005)
    
    threshold_90 = 0.90
    threshold_80 = 0.80
    def build_label(probs, label_flavour):
        return (
            fr"$\bf{label_flavour.latex}$ (N={len(probs)})" + "\n"
            + fr"frac>{threshold_90:.2f} = " + fr"$\mathbf{{{fraction_above_threshold(probs, threshold_90):.3f}}}$" + "\n"
            + fr"frac>{threshold_80:.2f} = " + fr"$\mathbf{{{fraction_above_threshold(probs, threshold_80):.3f}}}$"
        )

    label_e = build_label(nu_e_prob, Flavour.E)
    label_mu = build_label(nu_mu_prob, Flavour.MU)
    label_tau = build_label(nu_tau_prob, Flavour.TAU)
    
    fig, ax = plt.subplots(figsize=(17, 11))
    ax.hist(nu_e_prob, bins=bins, color=getColour(2), histtype='step', linewidth=1, label=label_e)
    ax.hist(nu_mu_prob, bins=bins, color=getColour(0), histtype='step', hatch='\\', linewidth=1, label=label_mu)
    ax.hist(nu_tau_prob, bins=bins, color=getColour(1), histtype='step', hatch ='//', linewidth=1, label=label_tau)
    ax.set_title(fr"{id}: ${flavour.latex}$ Probability Score", fontsize=24)
    ax.set_xlabel(fr"${flavour.latex}$ Probability Score", fontsize=24)
    ax.set_ylabel('Frequency')
    # located at the top middle
    ax.legend(fontsize=24, loc='upper center')
    plt.tight_layout()
    plt.show()
    
def get_significance(flavour, total_e, total_mu, total_tau):
    if flavour == Flavour.E:
        main_count = total_e
        opposite_count = total_mu + total_tau
    elif flavour == Flavour.MU:
        main_count = total_mu
        opposite_count = total_e + total_tau
    elif flavour == Flavour.TAU:
        main_count = total_tau
        opposite_count = total_e + total_mu
    
    poisson_label = fr"$Z_{{\mathrm{{Poisson}}, {flavour.latex}}}$"
    asimov_label = fr"$Z_{{\mathrm{{Asimov}}, {flavour.latex}}}$"

    poisson = main_count / np.sqrt(main_count + opposite_count) if (main_count + opposite_count) > 0 else 0
    asimov = np.sqrt(2*((main_count + opposite_count) * np.log(1 + main_count / (opposite_count + 1e-8)) - main_count))
    return poisson, asimov, poisson_label, asimov_label
    
def plot_prob_distribution_with_truncation(df: pd.DataFrame,
                                            flavour: Flavour,
                                            id: str,
                                            magnifier: Tuple[float, float] = None) -> None:
    nu_e_prob, nu_mu_prob, nu_tau_prob = get_nu_prob(df, flavour)
    Nbins, binwidth, bins, counts, bin_centers = getHistoParam(nu_tau_prob, binwidth=0.005)

    threshold_90 = 0.90
    threshold_80 = 0.80

    def build_label(probs, label_flavour):
        return (
            fr"$\bf{label_flavour.latex}$ (N={len(probs)})" + "\n"
            + fr"frac>{threshold_90:.2f} = " + fr"$\mathbf{{{fraction_above_threshold(probs, threshold_90):.3f}}}$" + "\n"
            + fr"frac>{threshold_80:.2f} = " + fr"$\mathbf{{{fraction_above_threshold(probs, threshold_80):.3f}}}$"
        )

    label_e = build_label(nu_e_prob, Flavour.E)
    label_mu = build_label(nu_mu_prob, Flavour.MU)
    label_tau = build_label(nu_tau_prob, Flavour.TAU)

    fig, ax = plt.subplots(figsize=(17, 11))
    n_e, _, _ = ax.hist(nu_e_prob, bins=bins, color=getColour(2), histtype='step',
                         linewidth=1, label=label_e)
    n_mu, _, _ = ax.hist(nu_mu_prob, bins=bins, color=getColour(0), histtype='step',
                          hatch='\\', linewidth=1, label=label_mu)
    n_tau, _, _ = ax.hist(nu_tau_prob, bins=bins, color=getColour(1), histtype='step',
                           hatch='//', linewidth=1, label=label_tau)
    truncate_at = 1000

    ymax = ax.get_ylim()[1]
    eps = 1e-6
    hist_stack = np.vstack([n_e, n_mu, n_tau])
    max_vals = hist_stack.max(axis=0)
    truncation_indices = np.where(max_vals >= (ymax - eps))[0]

    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    def get_union_annotation(indices):
        if indices.size == 0:
            return None, None
        start_bin = bins[indices.min()]
        end_bin = bins[min(indices.max() + 1, len(bins) - 1)]
        # Instead of bin_centres, directly build the index mask
        bin_mask = np.zeros_like(n_e, dtype=bool)
        bin_mask[indices.min() : indices.max() + 1] = True
        total_e = int(np.sum(n_e[bin_mask]))
        total_mu = int(np.sum(n_mu[bin_mask]))
        total_tau = int(np.sum(n_tau[bin_mask]))
        bin_range = f"{start_bin:.3f}–{end_bin:.3f}"
        
        poisson, asimov, poisson_label, asimov_label = get_significance(flavour, total_e, total_mu, total_tau)
        # significance, significance_label = get_significance(flavour, total_e, total_mu, total_tau)
        
        lines = [
            fr"${Flavour.E.latex}$ {bin_range} = {total_e}",
            fr"${Flavour.MU.latex}$ {bin_range} = {total_mu}",
            fr"${Flavour.TAU.latex}$ {bin_range} = {total_tau}",
            # fr"{significance_label} = {significance:.2f}"
            # fr"{poisson_label} = {poisson:.2f}",
            fr"{asimov_label} = {asimov:.2f}"
        ]
        return "\n".join(lines), bin_range
    
    def get_bin_annotation(bin_idx):
        start_bin = bins[bin_idx]
        end_bin = bins[min(bin_idx + 1, len(bins) - 1)]
        bin_range = f"{start_bin:.3f}–{end_bin:.3f}"
        total_e = int(n_e[bin_idx])
        total_mu = int(n_mu[bin_idx])
        total_tau = int(n_tau[bin_idx])
        
        poisson, asimov, poisson_label, asimov_label = get_significance(flavour, total_e, total_mu, total_tau)
        
        lines = [
            fr"${Flavour.E.latex}$ {bin_range} = {total_e}",
            fr"${Flavour.MU.latex}$ {bin_range} = {total_mu}",
            fr"${Flavour.TAU.latex}$ {bin_range} = {total_tau}",
            # fr"{poisson_label} = {poisson:.2f}",
            fr"{asimov_label} = {asimov:.2f}"
        ]
        return "\n".join(lines)
    
    if not magnifier:
        ax.set_ylim(0, truncate_at)
        ymax = truncate_at  # enforce limit as the "top"
        # Use slightly relaxed threshold
        truncation_indices = np.where(max_vals >= (ymax * 0.99))[0]

        # Low truncation zone
        low_indices = truncation_indices[bin_centers[truncation_indices] < 0.1]
        low_lines, _ = get_union_annotation(low_indices)
        if low_lines:
            ax.annotate(
                low_lines,
                xy=(0.10, 0.95),
                xycoords="axes fraction",
                fontsize=20,
                ha='left',
                va='top',
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.8)
            )

        # High truncation zone
        high_indices = truncation_indices[bin_centers[truncation_indices] > 0.9]
        high_lines, _ = get_union_annotation(high_indices)
        if high_lines:
            ax.annotate(
                high_lines,
                xy=(0.68, 0.95),
                xycoords="axes fraction",
                fontsize=24,
                ha='left',
                va='top',
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.8)
            )

    else:
        bin_mask = (bin_centers >= magnifier[0]) & (bin_centers <= magnifier[1])
        magnifier_bins = np.where(bin_mask)[0]
        if magnifier_bins.size > 0:
            # Find bin with maximum combined counts
            magnifier_max_bin_idx = magnifier_bins[np.argmax(hist_stack[:, magnifier_bins].sum(axis=0))]
            magnifier_lines = get_bin_annotation(magnifier_max_bin_idx)
            ax.annotate(
                magnifier_lines,
                xy=(0.68, 0.95),
                xycoords="axes fraction",
                fontsize=24,
                ha='left',
                va='top',
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.8)
            )

    ax.set_title(fr"{id}: ${flavour.latex}$ Probability Score")
    ax.set_xlabel(fr"${flavour.latex}$ Probability Score")
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=24, loc='upper center')
    if magnifier:
        ax.set_xlim(magnifier[0], magnifier[1])
    plt.tight_layout()
    plt.show()

def plot_log_energy_prob_distribution_mono_flavour(nu_prob : np.ndarray,
                                                    energy: np.ndarray,
                                                    output_flavour: Flavour,
                                                    flavour: Flavour, 
                                                    id: str,
                                                    normalise_by_energy: bool = True) -> None:
    Nbins, binwidth, bins, counts, bin_centers = getHistoParam(nu_prob, binwidth=0.1)
    log_energy_binwidth = 0.25
    log_energy = np.log10(energy)
    log_energy_bin = np.arange(np.min(log_energy), np.max(log_energy), log_energy_binwidth)

    # Swapped axes: Energy (x), Logit (y)
    raw_counts, xedges, yedges = np.histogram2d(log_energy, nu_prob, bins=[log_energy_bin, bins])
    H = raw_counts.copy()
    
    if normalise_by_energy:
        # Now energy is along x-axis, so normalise column-wise → axis=1
        H = H / (H.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(18, 12))
    mesh = ax.pcolormesh(xedges, yedges, H.T, cmap='YlGnBu', shading='auto')
    fig.colorbar(mesh, ax=ax, label="Frequency" if not normalise_by_energy else "Frequency normalised within energy bin")

    ax.set_title(fr"{id}: log₁₀(Energy [GeV])–Prob ${output_flavour.latex}$ of true ${flavour.latex}$", fontsize=24)
    ax.set_xlabel('log₁₀(Energy [GeV])', fontsize=24)
    ax.set_ylabel(fr'${flavour.latex}$ Probability Score', fontsize=24)
    # ax.set_ylim(-0.1, 1.1)

    ax.set_xlim(5, 8)

    # Annotate
    max_norm = np.max(H)
    xcentres = 0.5 * (xedges[:-1] + xedges[1:])
    ycentres = 0.5 * (yedges[:-1] + yedges[1:])
    for i in range(len(xcentres)):
        for j in range(len(ycentres)):
            count = raw_counts[i, j]
            norm = H[i, j]
            if count > 0:
                relative_intensity = norm / max_norm
                text_colour = 'white' if relative_intensity > 0.5 else 'black'
                ax.text(xcentres[i], ycentres[j], f"{norm:.2f}\n({int(count)})",
                        ha='center', va='center', fontsize=18, color=text_colour)
                
    plt.tight_layout()
    plt.show()

def plot_energy_distribution(df: pd.DataFrame, 
                                   flavour: Flavour,
                                   true_flavour: Flavour,
                                   id: str) -> None:
    nu_e_prob, nu_mu_prob, nu_tau_prob = get_nu_prob(df, flavour)
    e_energy, mu_energy, tau_energy = get_energy(df)
    if true_flavour == Flavour.E:
        plot_log_energy_prob_distribution_mono_flavour(nu_e_prob, e_energy, flavour, Flavour.E, id, normalise_by_energy=True)
    elif true_flavour == Flavour.MU:
        plot_log_energy_prob_distribution_mono_flavour(nu_mu_prob, mu_energy, flavour, Flavour.MU, id, normalise_by_energy=True)
    elif true_flavour == Flavour.TAU:
        plot_log_energy_prob_distribution_mono_flavour(nu_tau_prob, tau_energy, flavour, Flavour.TAU, id, normalise_by_energy=True)

def plot_zenith_prob_distribution_mono_flavour(nu_prob : np.ndarray,
                                                nu_zenith: np.ndarray,
                                                output_flavour: Flavour,
                                                flavour: Flavour, 
                                                id: str) -> None:
    Nbins, binwidth, bins, counts, bin_centers = getHistoParam(nu_prob, binwidth=0.1)
    zenith_binwidth = 20
    zenith_bin = np.arange(0, 180 + zenith_binwidth, zenith_binwidth)

    # Swap to zenith (x), Output (y)
    raw_counts, xedges, yedges = np.histogram2d(nu_zenith, nu_prob, bins=[zenith_bin, bins])
    H = raw_counts / (raw_counts.sum(axis=1, keepdims=True) + 1e-8)  # normalise by zenith bin (now rows)

    fig, ax = plt.subplots(figsize=(18, 12))
    mesh = ax.pcolormesh(xedges, yedges, H.T, cmap='YlOrBr', shading='auto')
    fig.colorbar(mesh, ax=ax, label="Counts normalised within zenith bin")

    ax.set_title(fr"{id}: Zenith–Prob ${output_flavour.latex}$ of true ${flavour.latex}$", fontsize=24)
    ax.set_xlabel('Zenith [degree]', fontsize=24)
    ax.set_ylabel(fr'${flavour.latex}$ Probability Score', fontsize=24)

    # Annotate bin centres
    max_norm = np.max(H)
    xcentres = 0.5 * (xedges[:-1] + xedges[1:])
    ycentres = 0.5 * (yedges[:-1] + yedges[1:])
    for i in range(len(xcentres)):
        for j in range(len(ycentres)):
            count = raw_counts[i, j]
            norm = H[i, j]
            if count > 0:
                relative_intensity = norm / max_norm
                text_colour = 'white' if relative_intensity > 0.5 else 'black'
                ax.text(xcentres[i], ycentres[j], f"{norm:.2f}\n({int(count)})",
                        ha='center', va='center', fontsize=18, color=text_colour)
    plt.tight_layout()
    plt.show()
                
def plot_zenith_distribution(df: pd.DataFrame, 
                                   flavour: Flavour,
                                   true_flavour: Flavour,
                                   id: str) -> None:
    nu_e_prob, nu_mu_prob, nu_tau_prob = get_nu_prob(df, flavour)
    e_zenith, mu_zenith, tau_zenith = get_zenith(df)
    
    if true_flavour == Flavour.E:
        plot_zenith_prob_distribution_mono_flavour(nu_e_prob, e_zenith, flavour, Flavour.E, id)
    elif true_flavour == Flavour.MU:
        plot_zenith_prob_distribution_mono_flavour(nu_mu_prob, mu_zenith, flavour, Flavour.MU, id)
    elif true_flavour == Flavour.TAU:
        plot_zenith_prob_distribution_mono_flavour(nu_tau_prob, tau_zenith, flavour, Flavour.TAU, id)

def extend_extract_metrics_for_all_flavours(df: pd.DataFrame, run_id: str, epoch: int) -> dict:
    # build id with run_id and epoch
    metrics = {"runID": run_id,
               "epoch": epoch}

    for true_flavour in Flavour:
        # Get prediction probabilities for each class on this true_flavour subset
        nu_e_prob, nu_mu_prob, nu_tau_prob = get_nu_prob(df, true_flavour)
        probs = {
            Flavour.E: nu_e_prob,
            Flavour.MU: nu_mu_prob,
            Flavour.TAU: nu_tau_prob,
        }

        target_prob = probs[true_flavour]
        rest = np.concatenate([probs[f] for f in Flavour if f != true_flavour])
        labels = np.concatenate([np.ones_like(target_prob), np.zeros_like(rest)])
        scores = np.concatenate([target_prob, rest])

        metrics.update({
            f"num_true_{true_flavour.alias}": len(target_prob),
            f"AUC_true_{true_flavour.alias}_vs_rest": roc_auc_score(labels, scores),
            f"frac_pred_{true_flavour.alias}_on_true_{true_flavour.alias}_prob_gt_0.9": fraction_above_threshold(target_prob, 0.9),
            f"frac_pred_{true_flavour.alias}_on_true_{true_flavour.alias}_prob_gt_0.8": fraction_above_threshold(target_prob, 0.8),
            f"frac_pred_{true_flavour.alias}_on_true_{true_flavour.alias}_prob_lt_0.2": fraction_below_threshold(target_prob, 0.2),
            f"frac_pred_{true_flavour.alias}_on_true_{true_flavour.alias}_prob_lt_0.1": fraction_below_threshold(target_prob, 0.1),
            f"median_pred_{true_flavour.alias}_on_true_{true_flavour.alias}_prob": float(np.median(target_prob)),
        })

        for pred_flavour in Flavour:
            if pred_flavour == true_flavour:
                continue
            pred_probs = probs[pred_flavour]
            metrics.update({
                f"frac_pred_{pred_flavour.alias}_on_true_{true_flavour.alias}_prob_gt_0.9": fraction_above_threshold(pred_probs, 0.9),
                f"frac_pred_{pred_flavour.alias}_on_true_{true_flavour.alias}_prob_gt_0.8": fraction_above_threshold(pred_probs, 0.8),
                f"median_pred_{pred_flavour.alias}_on_true_{true_flavour.alias}_prob": float(np.median(pred_probs)),
            })

    return metrics


def plot_all_metrics(df: pd.DataFrame, pdf_path: str, run_id: str, epoch: int) -> None:
    id = f"TrainID {run_id}," + f" epoch {epoch}"
    with PdfPages(pdf_path) as pdf:
        # Tau ROC
        plot_binary_flavour_ROC(df, Flavour.TAU, id)
        pdf.savefig()
        plt.close()
        
        plot_prob_distribution_with_truncation(df, Flavour.TAU, id)
        pdf.savefig()
        plt.close()

        # Tau Energy-Probability
        plot_energy_distribution(df, Flavour.TAU, Flavour.TAU, id)
        pdf.savefig()
        plt.close()

        # Tau Zenith-Probability
        plot_zenith_distribution(df, Flavour.TAU, Flavour.TAU, id)
        pdf.savefig()
        plt.close()
        
        # all flavours ROC
        plot_multi_flavour_ROC(df, id)
        pdf.savefig()
        plt.close()
        # leftovers: MU
        plot_binary_flavour_ROC(df, Flavour.MU, id)
        pdf.savefig()
        plt.close()
        
        plot_prob_distribution_with_truncation(df, Flavour.MU, id)
        pdf.savefig()
        plt.close()
        
        plot_energy_distribution(df, Flavour.MU, Flavour.MU, id)
        pdf.savefig()
        plt.close()
        
        plot_zenith_distribution(df, Flavour.MU, Flavour.MU, id)
        pdf.savefig()
        plt.close()
        
        # leftovers: E
        plot_binary_flavour_ROC(df, Flavour.E, id)
        pdf.savefig()
        plt.close()
        
        plot_prob_distribution_with_truncation(df, Flavour.E, id)
        pdf.savefig()
        plt.close()
        
        plot_energy_distribution(df, Flavour.E, Flavour.E, id)
        pdf.savefig()
        plt.close()
        
        plot_zenith_distribution(df, Flavour.E, Flavour.E, id)
        pdf.savefig()
        plt.close()