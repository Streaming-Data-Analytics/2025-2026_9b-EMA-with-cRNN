from sklearn.metrics import cohen_kappa_score
from scipy.stats import linregress
import numpy as np

#Compute the rolling Cohen's Kappa over a sliding window of (prediction, label) pairs.
#The kappa measures agreement between predictions and labels beyond chance, on a per-sample
#basis. The rolling window is reset immediately after every concept drift (when the task ID
#changes), so kappa values reflect only post-drift performance and are not contaminated by
#samples from the previous concept.
#Returns NaN at samples where the buffer is too small or contains only one class
def compute_rolling_kappa(preds, labels, tasks, window):
    kappas = []
    buf_preds, buf_labels = [], []
    prev_task = tasks[0]

    for i, (p, l, t) in enumerate(zip(preds, labels, tasks)):
        # Drift detected case: empty all previous predictions and actual labels
        if t != prev_task:
            buf_preds, buf_labels = [], []
            prev_task = t

        # Save current prediction and label
        buf_preds.append(p)
        buf_labels.append(l)

        # Keep only the samples of the last window
        if len(buf_preds) > window:
            buf_preds.pop(0)
            buf_labels.pop(0)

        # Need at least 2 distinct classes to compute kappa
        if len(set(buf_labels)) > 1:
            k = cohen_kappa_score(buf_labels, buf_preds)
        else:
            k = np.nan

        kappas.append(k)

    return np.array(kappas)

#Quantify EMA's average kappa advantage over the online model immediately after each drift.
#For each drift, computes the mean of (kappa_EMA - kappa_online) over the first `window`
#samples following the drift. Captures whether EMA's smoothed predictions are closer to
#the new concept's labels than the online model's noisier predictions during the
#immediate post-drift adaptation phase.
#positive = EMA better, negative = online better
def ema_advantage_per_drift(k_online, k_ema, tasks, window):
    drift_points = [i for i in range(1, len(tasks)) if tasks[i] != tasks[i-1]]

    results = []
    for d in drift_points:
        start = d
        end   = min(d + window, len(k_online))

        gap = k_ema[start:end] - k_online[start:end]
        gap = gap[np.isfinite(gap)]
        avg_gap = np.mean(gap) if len(gap) > 0 else np.nan
        results.append({
            'drift_at': d,
            'avg_gap': avg_gap,
            'ema_wins': avg_gap > 0
        })

    return results

#Compute the slope of the rolling kappa curve over a fixed initial window after a drift.
#Captures the immediate post-drift responsiveness — how steeply kappa rises (or falls)
#in the first samples following a concept change. The window length is fixed across all
#drifts so the resulting slopes are directly comparable; using a variable window
#(e.g., until threshold) would confound slope magnitude with window length.
def slope_initial_window(segment, window_length):
    """
    Compute slope of recovery over a fixed initial window after the drift.
    Uses the same window length for every drift to ensure slopes are comparable.
    Captures the immediate post-drift responsiveness.
    """
    segment = np.array(segment)
    end = min(window_length, len(segment))
    y = segment[:end]
    x = np.arange(len(y))

    valid = np.isfinite(y)

    if valid.sum() > 1:
        return linregress(x[valid], y[valid]).slope
    else:
        return np.nan

#Per-drift convergence speed analysis comparing the online and EMA models.
#For each concept drift, computes:
#- TTP (Time-To-Peak): the first sample after the drift where rolling kappa reaches
#   `ratio` × the shared maximum kappa achieved within that concept segment.
#   Both models are evaluated against the same target (the higher of their two maxima)
#   to ensure fair comparison.
#- Slope: regression slope of the kappa curve over the first slope_window samples
#   after the drift, measuring immediate responsiveness.
# The use of a relative threshold (ratio of within-concept maximum) instead of an
# absolute threshold makes the metric adaptive to concept difficulty: easy concepts
# where kappa rises high produce a high target, while harder concepts produce a low
# target, so the metric measures speed of adaptation rather than achievement of a
# universal performance level.
def convergence_speed_per_drift(k_online, k_ema, tasks, slope_window, ratio=0.9):
    """
    For each drift:
    - compute a relative threshold for each model
    - find TTP, the first step where the threshold is reached
    - compute slope from drift until TTP
    """

    drift_points = [i for i in range(1, len(tasks)) if tasks[i] != tasks[i-1]]
    results = []

    for d in drift_points:
        next_drifts = [x for x in drift_points if x > d]
        concept_end = next_drifts[0] if next_drifts else len(k_online)

        start = d
        end = concept_end

        seg_online = np.array(k_online[start:end])
        seg_ema = np.array(k_ema[start:end])

        valid_online = np.isfinite(seg_online)
        valid_ema = np.isfinite(seg_ema)

        # Use the same threshold for both models (the higher of the two)
        shared_max = max(np.nanmax(seg_online), np.nanmax(seg_ema))
        target_online = ratio * shared_max
        target_ema    = ratio * shared_max

        #target_online = ratio * np.nanmax(seg_online) if valid_online.sum() > 0 else np.nan
        #target_ema = ratio * np.nanmax(seg_ema) if valid_ema.sum() > 0 else np.nan

        hits_online = np.where(valid_online & (seg_online >= target_online))[0]
        hits_ema = np.where(valid_ema & (seg_ema >= target_ema))[0]

        ttp_online = int(hits_online[0]) if len(hits_online) > 0 else None
        ttp_ema = int(hits_ema[0]) if len(hits_ema) > 0 else None

        slope_online = slope_initial_window(seg_online, slope_window)
        slope_ema    = slope_initial_window(seg_ema,    slope_window)

        results.append({
            "drift_at": d,
            "concept_from": int(tasks[d - 1]),
            "concept_to": int(tasks[d]),

            "target_online": target_online,
            "target_ema": target_ema,

            "ttp_online": ttp_online,
            "ttp_ema": ttp_ema,
            "ttp_ema_wins": (
                (ttp_ema is not None and ttp_online is not None and ttp_ema < ttp_online)
                or (ttp_ema is not None and ttp_online is None)
            ),

            "slope_online": slope_online,
            "slope_ema": slope_ema,
            "slope_ema_wins": slope_ema > slope_online,
        })


    return results
