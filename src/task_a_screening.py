"""Held-out building-type screening."""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .config import BUILDING_TYPES, FEATURE_NAMES, TaskAConfig
logger = logging.getLogger(__name__)
__all__ = ['TaskAResult', 'run_task_a']

@dataclass
class TaskAResult:
    """Public class: TaskAResult."""
    silhouettes: Dict[int, float]
    k_silhouette_optimal: int
    k_selected: int
    confusion_train: np.ndarray
    confusion_test: np.ndarray
    per_class_test: pd.DataFrame
    accuracy_test: float
    balanced_accuracy_test: float
    macro_f1_test: float
    weighted_f1_test: float
    accuracy_train: float
    ari_bootstrap: Tuple[float, float]
    labels: List[str]
    predicted_type: pd.Series
    is_test: pd.Series
    binary_test: Dict[str, float]
    occupied_probability: pd.Series

def _cluster_to_type(features: pd.DataFrame, clusters: np.ndarray, true_types: np.ndarray, k: int) -> Dict[int, str]:
    """Public function: _cluster_to_type."""
    mapping: Dict[int, str] = {}
    for c in range(k):
        members = true_types[clusters == c]
        if len(members) == 0:
            mapping[c] = 'residential'
            continue
        (vals, counts) = np.unique(members, return_counts=True)
        mapping[c] = str(vals[np.argmax(counts)])
    return mapping

def run_task_a(features: pd.DataFrame, meta: pd.DataFrame, cfg: Optional[TaskAConfig]=None) -> TaskAResult:
    """Public function: run_task_a."""
    cfg = cfg or TaskAConfig()
    X = features[list(FEATURE_NAMES)].values
    y = meta.sort_values('household_id')['true_type'].values
    hids = features['household_id'].values
    (idx_tr, idx_te) = train_test_split(np.arange(len(y)), test_size=cfg.test_size, random_state=cfg.seed, stratify=y)
    scaler = StandardScaler().fit(X[idx_tr])
    X_tr = scaler.transform(X[idx_tr])
    X_te = scaler.transform(X[idx_te])
    silhouettes: Dict[int, float] = {}
    for k in cfg.k_candidates:
        km = KMeans(n_clusters=k, random_state=cfg.seed, n_init=cfg.n_init)
        lab = km.fit_predict(X_tr)
        silhouettes[k] = float(silhouette_score(X_tr, lab))
    k_opt = max(silhouettes, key=silhouettes.get)
    logger.info('silhouette profile: %s -> optimum K=%d', {k: round(v, 4) for (k, v) in silhouettes.items()}, k_opt)
    k_selected = int(k_opt if cfg.k_selected is None else cfg.k_selected)
    km = KMeans(n_clusters=k_selected, random_state=cfg.seed, n_init=cfg.n_init)
    clu_tr = km.fit_predict(X_tr)
    mapping = _cluster_to_type(features.iloc[idx_tr], clu_tr, y[idx_tr], k_selected)
    clu_te = km.predict(X_te)
    pred_tr = np.array([mapping[c] for c in clu_tr])
    pred_te = np.array([mapping[c] for c in clu_te])
    occupied = {'residential', 'mixed'}
    y_bin_train = np.isin(y[idx_tr], list(occupied)).astype(int)
    gate_calibrator = LogisticRegression(C=1.0, max_iter=1000, random_state=cfg.seed).fit(km.transform(X_tr), y_bin_train)
    occupied_probability_values = gate_calibrator.predict_proba(km.transform(scaler.transform(X)))[:, 1]
    labels = list(BUILDING_TYPES)
    cm_tr = confusion_matrix(y[idx_tr], pred_tr, labels=labels)
    cm_te = confusion_matrix(y[idx_te], pred_te, labels=labels)
    (p, r, f1, sup) = precision_recall_fscore_support(y[idx_te], pred_te, labels=labels, zero_division=0)
    per_class = pd.DataFrame({'type': labels, 'precision': p, 'recall': r, 'f1': f1, 'support': sup})
    rng = np.random.RandomState(cfg.seed)
    aris: List[float] = []
    for _ in range(cfg.n_bootstrap):
        boot = rng.choice(len(idx_tr), len(idx_tr), replace=True)
        km_b = KMeans(n_clusters=k_selected, random_state=rng.randint(1 << 30), n_init=cfg.n_init).fit(X_tr[boot])
        aris.append(float(adjusted_rand_score(clu_tr[boot], km_b.labels_)))
    (ari_mean, ari_sd) = (float(np.mean(aris)), float(np.std(aris)))
    y_bin = np.isin(y[idx_te], list(occupied))
    p_bin = np.isin(pred_te, list(occupied))
    tp = int(np.sum(y_bin & p_bin))
    fp = int(np.sum(~y_bin & p_bin))
    fn = int(np.sum(y_bin & ~p_bin))
    tn = int(np.sum(~y_bin & ~p_bin))
    binary = {'accuracy': float((tp + tn) / max(1, tp + tn + fp + fn)), 'precision': float(tp / max(1, tp + fp)), 'recall': float(tp / max(1, tp + fn)), 'f1': float(2 * tp / max(1, 2 * tp + fp + fn)), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}
    full_pred = pd.Series(index=pd.Index(hids, name='household_id'), dtype=object)
    full_pred.iloc[idx_tr] = pred_tr
    full_pred.iloc[idx_te] = pred_te
    is_test = pd.Series(False, index=pd.Index(hids, name='household_id'))
    is_test.iloc[idx_te] = True
    occupied_probability = pd.Series(occupied_probability_values, index=pd.Index(hids, name='household_id'), dtype=float)
    result = TaskAResult(silhouettes=silhouettes, k_silhouette_optimal=int(k_opt), k_selected=k_selected, confusion_train=cm_tr, confusion_test=cm_te, per_class_test=per_class, accuracy_test=float(np.mean(pred_te == y[idx_te])), balanced_accuracy_test=float(balanced_accuracy_score(y[idx_te], pred_te)), macro_f1_test=float(f1_score(y[idx_te], pred_te, average='macro', zero_division=0)), weighted_f1_test=float(f1_score(y[idx_te], pred_te, average='weighted', zero_division=0)), accuracy_train=float(np.mean(pred_tr == y[idx_tr])), ari_bootstrap=(ari_mean, ari_sd), labels=labels, predicted_type=full_pred, is_test=is_test, binary_test=binary, occupied_probability=occupied_probability)
    logger.info('Task A held-out: acc=%.4f bal_acc=%.4f macro_F1=%.4f | occupied-screen F1=%.4f', result.accuracy_test, result.balanced_accuracy_test, result.macro_f1_test, binary['f1'])
    return result
