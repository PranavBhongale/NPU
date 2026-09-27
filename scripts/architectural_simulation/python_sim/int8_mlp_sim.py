import matplotlib
matplotlib.use("Agg")   # headless-safe backend, saves to file instead of showing a window
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(123)

S_STAR = 3.9150          # s* in units of sigma (Gaussian-optimal, from Part 1)
LEVELS = 256


def plot_mse_vs_clip_range(t, s_star=S_STAR, num_points=200, save_path="mse_vs_clip.png",
                            tensor_label="tensor", rel_error_threshold_pct=1.0,
                            s_range=(1.0, 6.0), auto_extend=True):
    """
    Plot INT8 quantization error vs clipping range, with the information
    needed to argue INT8 is low-error annotated directly on the figure:

      - SQNR (dB) at the optimum, on a secondary y-axis across the whole curve
      - RMS relative error (%) at the optimum
      - Effective number of bits (ENOB) actually achieved out of 8
      - A reference line at a chosen "acceptable error" threshold (default 1%)
        so the optimum's position relative to that bar is visible at a glance

    s_range: (lo, hi) search window in units of sigma. If auto_extend is True
    and the found minimum lands on the right edge of the window, the window
    is doubled and re-searched -- this catches skewed/heavy-tailed tensors
    (e.g. post-ReLU activations) whose true optimal clip lies further out
    than a Gaussian-tuned default window would show, which would otherwise
    silently report a boundary artifact as if it were the true optimum.
    """
    sigma = t.std()
    lo, hi = s_range

    for _ in range(6):  # bounded retries, avoid infinite loop on pathological data
        s_values = np.linspace(lo, hi, num_points)
        mse_values = np.empty_like(s_values)
        for i, s_norm in enumerate(s_values):
            alpha = s_norm * sigma
            delta = (2 * alpha) / (LEVELS - 1)
            tq = np.clip(t, -alpha, alpha)
            tq = np.round(tq / delta) * delta
            mse_values[i] = np.mean((t - tq) ** 2) / (sigma ** 2)

        min_index = np.argmin(mse_values)
        at_right_edge = min_index >= num_points - 2
        if at_right_edge and auto_extend:
            print(f"[warning] optimum landed at search-range edge (hi={hi:.1f}σ) "
                  f"for '{tensor_label}' -- widening search range and retrying")
            lo, hi = hi * 0.5, hi * 2.0
            continue
        break

    optimal_s = s_values[min_index]
    minimum_mse = mse_values[min_index]

    # ---- derived, human-readable error metrics at the optimum ----
    sqnr_db = 10 * np.log10(1.0 / minimum_mse)          # signal power = 1 in normalized units
    rms_rel_error_pct = np.sqrt(minimum_mse) * 100       # RMS error as % of sigma
    enob = (sqnr_db - 1.76) / 6.02                       # effective bits actually achieved

    # SQNR curve across the whole sweep (for the secondary axis)
    sqnr_curve_db = 10 * np.log10(1.0 / mse_values)

    fig, ax1 = plt.subplots(figsize=(9, 6))

    # --- primary axis: normalized MSE, log scale so the curve is legible
    #     across the steep rounding-dominated and clipping-dominated regions
    ax1.plot(s_values, mse_values, color="tab:blue", linewidth=2, label="Normalized MSE")
    ax1.set_yscale("log")
    ax1.set_xlabel("Clipping range α / σ")
    ax1.set_ylabel("Normalized MSE (log scale)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, which="both", alpha=0.3)

    # optimum marker
    ax1.scatter([optimal_s], [minimum_mse], color="tab:red", s=90, zorder=5,
                label=f"Optimum: {optimal_s:.3f}σ")
    ax1.axvline(optimal_s, color="tab:red", linestyle="--", alpha=0.5)

    # --- secondary axis: SQNR in dB (this is the number that actually
    #     communicates "how much error" in an intuitive, log-perceptual way)
    ax2 = ax1.twinx()
    ax2.plot(s_values, sqnr_curve_db, color="tab:green", linewidth=1.5, alpha=0.7,
              linestyle=":", label="SQNR (dB)")
    ax2.set_ylabel("SQNR (dB)", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    # reference line: the RMS-relative-error threshold you consider "acceptable"
    threshold_mse = (rel_error_threshold_pct / 100.0) ** 2
    ax1.axhline(threshold_mse, color="gray", linestyle="--", alpha=0.6)
    ax1.text(s_values[-1], threshold_mse, f"  {rel_error_threshold_pct:.1f}% RMS error line",
              va="bottom", ha="right", fontsize=8, color="gray")

    # --- annotation box with the claim-supporting numbers ---
    info_text = (
        f"Optimal clip:      {optimal_s:.3f} σ\n"
        f"Min normalized MSE: {minimum_mse:.3e}\n"
        f"SQNR:              {sqnr_db:.2f} dB\n"
        f"RMS error:          {rms_rel_error_pct:.3f}% of σ\n"
        f"Effective bits:     {enob:.2f} / 8 (ENOB)"
    )
    ax1.text(0.03, 0.06, info_text, transform=ax1.transAxes, fontsize=10,
              family="monospace", va="bottom", ha="left",
              bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))

    ax1.set_title(f"INT8 Quantization Error vs Clipping Range ({tensor_label})")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print("\nINT8 MSE Optimization")
    print("---------------------")
    print(f"Optimal clipping range = {optimal_s:.4f} sigma")
    print(f"Minimum normalized MSE = {minimum_mse:.6e}")
    print(f"SQNR                   = {sqnr_db:.2f} dB")
    print(f"RMS relative error     = {rms_rel_error_pct:.3f}% of sigma")
    print(f"Effective bits (ENOB)  = {enob:.2f} / 8")
    print(f"Saved plot to: {save_path}")

    return optimal_s, minimum_mse, sqnr_db, rms_rel_error_pct, enob


def int8_quantize(t, s_star=S_STAR):
    sigma = t.std()
    alpha = s_star * sigma
    delta = (2 * alpha) / (LEVELS - 1)
    tq = np.clip(t, -alpha, alpha)
    tq = np.round(tq / delta) * delta
    mse = float(np.mean((t - tq) ** 2))
    sqnr = 10 * np.log10((t.var() + 1e-12) / (mse + 1e-12))
    return tq, mse, sqnr


def relu(x):
    return np.maximum(0, x)


# =========================================================
# Build a small 3-layer MLP classifier (FP32 "golden" model)
# =========================================================
d_in, h1, h2, K = 128, 256, 128, 10
N = 30_000

W1 = rng.normal(0, 1 / np.sqrt(d_in), size=(d_in, h1))
W2 = rng.normal(0, 1 / np.sqrt(h1), size=(h1, h2))
W3 = rng.normal(0, 1 / np.sqrt(h2), size=(h2, K))
b1 = rng.normal(0, 0.05, size=h1)
b2 = rng.normal(0, 0.05, size=h2)
b3 = rng.normal(0, 0.05, size=K)

X = rng.normal(0, 1.0, size=(N, d_in))


def forward(X, W1, b1, W2, b2, W3, b3):
    a1_pre = X @ W1 + b1
    a1 = relu(a1_pre)
    a2_pre = a1 @ W2 + b2
    a2 = relu(a2_pre)
    z = a2 @ W3 + b3
    return a1_pre, a1, a2_pre, a2, z


a1_pre, a1, a2_pre, a2, Z = forward(X, W1, b1, W2, b2, W3, b3)
y_pred = np.argmax(Z, axis=1)

W1q, mse_W1, sqnr_W1 = int8_quantize(W1)
W2q, mse_W2, sqnr_W2 = int8_quantize(W2)
W3q, mse_W3, sqnr_W3 = int8_quantize(W3)
A1q, mse_A1, sqnr_A1 = int8_quantize(a1)
A2q, mse_A2, sqnr_A2 = int8_quantize(a2)

print("=" * 70)
print("PER-LAYER INT8 QUANTIZATION TABLE (weight-optimal s* applied)")
print("=" * 70)
rows = [
    ("Layer1", "Weight", mse_W1, sqnr_W1),
    ("Layer1", "Activation", mse_A1, sqnr_A1),
    ("Layer2", "Weight", mse_W2, sqnr_W2),
    ("Layer2", "Activation", mse_A2, sqnr_A2),
    ("Layer3", "Weight", mse_W3, sqnr_W3),
]
print(f"{'Layer':<8}{'Tensor':<12}{'MSE':>14}{'SQNR (dB)':>12}")
for layer, tensor, mse, sqnr in rows:
    print(f"{layer:<8}{tensor:<12}{mse:>14.3e}{sqnr:>12.2f}")
print()

Xq, _, _ = int8_quantize(X)
a1_pre_q = Xq @ W1q + b1
a1_q_full = relu(a1_pre_q)
a1_q_full, _, _ = int8_quantize(a1_q_full)
a2_pre_q = a1_q_full @ W2q + b2
a2_q_full = relu(a2_pre_q)
a2_q_full, _, _ = int8_quantize(a2_q_full)
Zq = a2_q_full @ W3q + b3
y_pred_q = np.argmax(Zq, axis=1)

sorted_z = np.sort(Z, axis=1)
top1, top2 = sorted_z[:, -1], sorted_z[:, -2]
margin = top1 - top2
delta_z = Zq - Z
eps = np.max(np.abs(delta_z), axis=1)
theory_preserved = margin > 2 * eps
actual_preserved = (y_pred_q == y_pred)

print("=" * 70)
print("END-TO-END 3-LAYER MLP, FULL INT8 DATAPATH (weights + activations)")
print("=" * 70)
print(f"Validation samples                         = {N}")
print(f"Theoretical guarantee  (m_i > 2*eps_i)      = {theory_preserved.mean()*100:.3f}%")
print(f"Actual prediction match (empirical)         = {actual_preserved.mean()*100:.3f}%")
print(f"Guarantee is a valid subset of actual: "
      f"{np.all(actual_preserved[theory_preserved])}")
print()

# Generate the annotated plot for the input tensor X (Gaussian, sigma=1)
plot_mse_vs_clip_range(X, tensor_label="Input activations X (Gaussian)",
                        save_path="mse_vs_clip_X.png")

# Also generate one for the ReLU activation a1, to visually contrast
# the non-Gaussian case discussed earlier
plot_mse_vs_clip_range(a1, tensor_label="Layer-1 post-ReLU activations",
                        save_path="mse_vs_clip_a1.png")