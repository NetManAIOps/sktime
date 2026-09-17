# -*- coding:utf-8 -*-
import numpy as np
from sklearn import metrics
from typing import Iterable, List, Sequence, Tuple

from .affiliation import convert_vector_to_events, pr_from_events

Range = Tuple[int, int]


class base_metricor:

    METRIC_MAP = {
        "AUC-ROC": {
            "method": "metric_AUC_ROC",
            "requires": ("label", "score"),
            "params": {},
        },
        "AP": {
            "method": "metric_AP",
            "requires": ("label", "score"),
            "params": {},
        },
        "Point-F1": {
            "method": "metric_PointF1",
            "requires": ("label", "score"),
            "params": {},
        },
        "PA-F1": {
            "method": "metric_PointF1PA",
            "requires": ("label", "score"),
            "params": {},
        },
        "Affiliation-F1": {
            "method": "metric_AffiliationF1",
            "requires": ("label", "score"),
            "params": {},
        },
        "Delay-F1": {
            "method": "metric_PointF1PA_delay_k",
            "requires": ("label", "score"),
            "params": {
                "delay_k": 7,
            },
        },
        "VUS-PR": {
            "method": "metric_vus_pr",
            "requires": ("label", "score"),
            "params": {
                "sliding_window": 100,
            },
        },
        "VUS-ROC": {
            "method": "metric_vus_roc",
            "requires": ("label", "score"),
            "params": {
                "sliding_window": 100,
            },
        },
        "Predict-Error": {
            "method": "metric_output_mse",
            "requires": ("values", "output"),
            "params": {},
        },
        "Output-MAE": {
            "method": "metric_output_mae",
            "requires": ("values", "output"),
            "params": {},
        },
    }

    def _adjust_predicts(self, label, pred, calc_latency=False):
        """对二值预测pred做point-adjustment"""
        label = np.asarray(label).astype(int).squeeze()
        pred  = np.asarray(pred).astype(int).squeeze()

        if len(pred) != len(label):
            raise ValueError("pred and label must have the same length")

        actual  = label.astype(bool)
        predict = pred.astype(bool).copy()

        latency = 0
        anomaly_state = False
        anomaly_count = 0

        for i in range(len(predict)):
            if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, -1, -1):
                    if not actual[j]:
                        break
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
            elif not actual[i]:
                anomaly_state = False

            if anomaly_state:
                predict[i] = True

        if calc_latency:
            return predict.astype(int), latency / (anomaly_count + 1e-4)
        else:
            return predict.astype(int)

    def _adjust_predicts_delay_k(self, label, pred, delay_k: int):
        """Point-adjustment with delay constraint (only count hits within k)."""
        label = np.asarray(label).astype(int).squeeze()
        pred = np.asarray(pred).astype(int).squeeze()

        if len(pred) != len(label):
            raise ValueError("pred and label must have the same length")

        n = len(label)
        predict = pred.astype(bool).copy()
        actual = label.astype(bool)
        delay_k = int(delay_k)
        if delay_k < 0:
            delay_k = 0

        i = 0
        while i < n:
            if not actual[i]:
                i += 1
                continue

            start = i
            while i < n and actual[i]:
                i += 1
            end = i

            hit_end = min(end, start + delay_k + 1)
            hit = predict[start:hit_end].any()

            if hit:
                predict[start:end] = True

        return predict.astype(int)
        
    # -------- VUS-PR core  --------
    def range_convers_new(self, label: np.ndarray) -> List[Range]:
        """Convert a binary label vector into a list of anomaly ranges."""
        anomaly_starts = np.where(np.diff(label) == 1)[0] + 1
        anomaly_ends, = np.where(np.diff(label) == -1)

        if len(anomaly_ends):
            if not len(anomaly_starts) or anomaly_ends[0] < anomaly_starts[0]:
                anomaly_starts = np.concatenate([[0], anomaly_starts])
        if len(anomaly_starts):
            if not len(anomaly_ends) or anomaly_ends[-1] < anomaly_starts[-1]:
                anomaly_ends = np.concatenate([anomaly_ends, [len(label) - 1]])

        return list(zip(anomaly_starts, anomaly_ends))

    def new_sequence(self, label: np.ndarray, sequence_original: Sequence[Range], window: int) -> List[Range]:
        """Extend anomaly segments by half a window and merge nearby segments."""
        if not sequence_original:
            return []

        a = max(sequence_original[0][0] - window // 2, 0)
        sequence_new: List[Range] = []

        for i in range(len(sequence_original) - 1):
            if sequence_original[i][1] + window // 2 < sequence_original[i + 1][0] - window // 2:
                sequence_new.append((a, sequence_original[i][1] + window // 2))
                a = sequence_original[i + 1][0] - window // 2

        last_end = min(sequence_original[-1][1] + window // 2, len(label) - 1)
        sequence_new.append((a, last_end))
        return sequence_new

    def sequencing(self, x: np.ndarray, ranges: Sequence[Range], window: int = 5) -> np.ndarray:
        """Smoothly extend labels around anomaly ranges using the repository rule."""
        label = x.copy().astype(float)
        length = len(label)

        if window <= 0:
            return label

        for s, e in ranges:
            x1 = np.arange(e + 1, min(e + window // 2 + 1, length))
            label[x1] += np.sqrt(1 - (x1 - e) / window)

            x2 = np.arange(max(s - window // 2, 0), s)
            label[x2] += np.sqrt(1 - (s - x2) / window)

        label = np.minimum(np.ones(length), label)
        return label

    def RangeAUC_volume_opt(self, labels_original: np.ndarray, score: np.ndarray, windowSize: int, thre: int = 250):
        """Direct extraction of ``RangeAUC_volume_opt`` from ``basic_metricor``."""
        window_3d = np.arange(0, windowSize + 1, 1)
        P = np.sum(labels_original)
        seq = self.range_convers_new(labels_original)
        l = self.new_sequence(labels_original, seq, windowSize)

        if len(score) == 0:
            raise ValueError("score must be non-empty")

        score_sorted = -np.sort(-score)
        threshold_indices = np.linspace(0, len(score) - 1, thre).astype(int)

        tpr_3d = np.zeros((windowSize + 1, thre + 2))
        fpr_3d = np.zeros((windowSize + 1, thre + 2))
        prec_3d = np.zeros((windowSize + 1, thre + 1))

        auc_3d = np.zeros(windowSize + 1)
        ap_3d = np.zeros(windowSize + 1)

        tp = np.zeros(thre)
        N_pred = np.zeros(thre)

        for k, i in enumerate(threshold_indices):
            threshold = score_sorted[i]
            pred = score >= threshold
            N_pred[k] = np.sum(pred)

        for window in window_3d:
            labels_extended = self.sequencing(labels_original, seq, window)
            L = self.new_sequence(labels_extended, seq, window)

            if not L or not l:
                continue

            TF_list = np.zeros((thre + 2, 2))
            Precision_list = np.ones(thre + 1)
            j = 0

            for i in threshold_indices:
                threshold = score_sorted[i]
                pred = score >= threshold
                labels = labels_extended.copy()
                existence = 0

                for seg in L:
                    labels[seg[0]:seg[1] + 1] = labels_extended[seg[0]:seg[1] + 1] * pred[seg[0]:seg[1] + 1]
                    if (pred[seg[0]:seg[1] + 1] > 0).any():
                        existence += 1
                for seg in seq:
                    labels[seg[0]:seg[1] + 1] = 1

                TP = 0.0
                N_labels = 0.0
                for seg in l:
                    TP += np.dot(labels[seg[0]:seg[1] + 1], pred[seg[0]:seg[1] + 1])
                    N_labels += np.sum(labels[seg[0]:seg[1] + 1])

                TP += tp[j]
                FP = N_pred[j] - TP

                existence_ratio = existence / len(L)
                P_new = (P + N_labels) / 2
                recall = min(TP / P_new, 1)

                TPR = recall * existence_ratio
                N_new = len(labels) - P_new
                FPR = FP / N_new

                Precision = TP / N_pred[j] if N_pred[j] > 0 else 0.0

                j += 1
                TF_list[j] = [TPR, FPR]
                Precision_list[j] = Precision

            TF_list[j + 1] = [1, 1]

            tpr_3d[window] = TF_list[:, 0]
            fpr_3d[window] = TF_list[:, 1]
            prec_3d[window] = Precision_list

            width = TF_list[1:, 1] - TF_list[:-1, 1]
            height = (TF_list[1:, 0] + TF_list[:-1, 0]) / 2
            auc_3d[window] = np.dot(width, height)

            width_PR = TF_list[1:-1, 0] - TF_list[:-2, 0]
            height_PR = Precision_list[1:]
            ap_3d[window] = np.dot(width_PR, height_PR)

        return tpr_3d, fpr_3d, prec_3d, window_3d, float(np.mean(auc_3d)), float(np.mean(ap_3d))

    def RangeAUC_volume_opt_mem(self, labels_original: np.ndarray, score: np.ndarray, windowSize: int, thre: int = 250):
        """Direct extraction of ``RangeAUC_volume_opt_mem`` from ``basic_metricor``."""
        window_3d = np.arange(0, windowSize + 1, 1)
        P = np.sum(labels_original)
        seq = self.range_convers_new(labels_original)
        l = self.new_sequence(labels_original, seq, windowSize)

        if len(score) == 0:
            raise ValueError("score must be non-empty")

        score_sorted = -np.sort(-score)
        threshold_indices = np.linspace(0, len(score) - 1, thre).astype(int)

        tpr_3d = np.zeros((windowSize + 1, thre + 2))
        fpr_3d = np.zeros((windowSize + 1, thre + 2))
        prec_3d = np.zeros((windowSize + 1, thre + 1))

        auc_3d = np.zeros(windowSize + 1)
        ap_3d = np.zeros(windowSize + 1)

        tp = np.zeros(thre)
        N_pred = np.zeros(thre)
        p = np.zeros((thre, len(score)))

        for k, i in enumerate(threshold_indices):
            threshold = score_sorted[i]
            pred = score >= threshold
            p[k] = pred
            N_pred[k] = np.sum(pred)

        for window in window_3d:
            labels_extended = self.sequencing(labels_original, seq, window)
            L = self.new_sequence(labels_extended, seq, window)

            if not L or not l:
                continue

            TF_list = np.zeros((thre + 2, 2))
            Precision_list = np.ones(thre + 1)
            j = 0

            for _ in threshold_indices:
                labels = labels_extended.copy()
                existence = 0

                for seg in L:
                    labels[seg[0]:seg[1] + 1] = labels_extended[seg[0]:seg[1] + 1] * p[j][seg[0]:seg[1] + 1]
                    if (p[j][seg[0]:seg[1] + 1] > 0).any():
                        existence += 1
                for seg in seq:
                    labels[seg[0]:seg[1] + 1] = 1

                N_labels = 0.0
                TP = 0.0
                for seg in l:
                    TP += np.dot(labels[seg[0]:seg[1] + 1], p[j][seg[0]:seg[1] + 1])
                    N_labels += np.sum(labels[seg[0]:seg[1] + 1])

                TP += tp[j]
                FP = N_pred[j] - TP

                existence_ratio = existence / len(L)
                P_new = (P + N_labels) / 2
                recall = min(TP / P_new, 1)

                TPR = recall * existence_ratio
                N_new = len(labels) - P_new
                FPR = FP / N_new
                Precision = TP / N_pred[j] if N_pred[j] > 0 else 0.0
                j += 1

                TF_list[j] = [TPR, FPR]
                Precision_list[j] = Precision

            TF_list[j + 1] = [1, 1]
            tpr_3d[window] = TF_list[:, 0]
            fpr_3d[window] = TF_list[:, 1]
            prec_3d[window] = Precision_list

            width = TF_list[1:, 1] - TF_list[:-1, 1]
            height = (TF_list[1:, 0] + TF_list[:-1, 0]) / 2
            auc_3d[window] = np.dot(width, height)

            width_PR = TF_list[1:-1, 0] - TF_list[:-2, 0]
            height_PR = Precision_list[1:]
            ap_3d[window] = np.dot(width_PR, height_PR)
        return tpr_3d, fpr_3d, prec_3d, window_3d, float(np.mean(auc_3d)), float(np.mean(ap_3d))
    
    # -------- Metrics  --------
    def metric_AUC_ROC(self, label, score):
        label = np.asarray(label).astype(int).squeeze()
        score = np.asarray(score).astype(float).squeeze()
        return metrics.roc_auc_score(label, score)

    def metric_AP(self, label, score):  # AP
        label = np.asarray(label).astype(int).squeeze()
        score = np.asarray(score).astype(float).squeeze()
        return metrics.average_precision_score(label, score)

    def metric_output_mse(self, values, output):
        values = np.asarray(values).astype(float).squeeze()
        output = np.asarray(output).astype(float).squeeze()
        return np.mean(np.square(values - output))

    def metric_output_mae(self, values, output):
        values = np.asarray(values).astype(float).squeeze()
        output = np.asarray(output).astype(float).squeeze()
        return np.mean(np.abs(values - output))

    def metric_PointF1(self, label, score, preds=None):
        label = np.asarray(label).astype(int).squeeze()
        score = np.asarray(score).astype(float).squeeze()
        if preds is None:
            precision, recall, _ = metrics.precision_recall_curve(label, score)
            f1_scores = 2 * (precision * recall) / (precision + recall + 0.00001)
            F1 = np.max(f1_scores)
        else:
            preds = np.asarray(preds).astype(int).squeeze()
            _, _, F, _ = metrics.precision_recall_fscore_support(label, preds, zero_division=0)
            F1 = F[1]
        return F1
    
    def metric_PointF1PA(self, label, score, preds=None, max_thresholds=600):
        label = np.asarray(label).astype(int).squeeze()
        score = np.asarray(score).astype(float).squeeze()

        if preds is None:
            thresholds = np.unique(score)
            if thresholds.size > max_thresholds:
                qs = np.linspace(0, 1, max_thresholds)
                thresholds = np.quantile(score, qs)
                thresholds = np.unique(thresholds)
            best = -1.0
            for th in thresholds:
                y_pred = (score >= th).astype(int)
                y_adj = self._adjust_predicts(label, y_pred)
                best = max(best, metrics.f1_score(label, y_adj))
            return best
        else:
            y_pred = np.asarray(preds).astype(int).squeeze()
            y_adj = self._adjust_predicts(label, y_pred)
            return metrics.f1_score(label, y_adj)

    def metric_PointF1PA_delay_k(self, label, score, delay_k: int = 7, preds=None, max_thresholds=600):
        label = np.asarray(label).astype(int).squeeze()
        score = np.asarray(score).astype(float).squeeze()

        if preds is None:
            thresholds = np.unique(score)
            if thresholds.size > max_thresholds:
                qs = np.linspace(0, 1, max_thresholds)
                thresholds = np.quantile(score, qs)
                thresholds = np.unique(thresholds)
            best = -1.0
            for th in thresholds:
                y_pred = (score >= th).astype(int)
                y_adj = self._adjust_predicts_delay_k(label, y_pred, delay_k=delay_k)
                best = max(best, metrics.f1_score(label, y_adj))
            return best
        else:
            y_pred = np.asarray(preds).astype(int).squeeze()
            y_adj = self._adjust_predicts_delay_k(label, y_pred, delay_k=delay_k)
            return metrics.f1_score(label, y_adj)

    def metric_AffiliationF1(self, label, score, preds=None, max_thresholds=600):
        label = np.asarray(label).astype(int).squeeze()
        score = np.asarray(score).astype(float).squeeze()

        if len(label) != len(score):
            raise ValueError("label and score must have the same length")

        events_gt = convert_vector_to_events(label)
        if not events_gt:
            raise ValueError("Affiliation-F1 requires at least one anomaly event")

        def affiliation_f1(y_pred: np.ndarray) -> float:
            events_pred = convert_vector_to_events(y_pred)
            if not events_pred:
                return 0.0

            result = pr_from_events(
                events_pred=events_pred,
                events_gt=events_gt,
                Trange=(0, len(label)),
            )
            precision = result["precision"]
            recall = result["recall"]
            denominator = precision + recall
            if not np.isfinite(denominator) or denominator == 0:
                return 0.0
            return 2 * precision * recall / denominator

        if preds is not None:
            y_pred = np.asarray(preds).astype(int).squeeze()
            if len(y_pred) != len(label):
                raise ValueError("preds and label must have the same length")
            return affiliation_f1(y_pred)

        thresholds = np.unique(score)
        if thresholds.size > max_thresholds:
            qs = np.linspace(0, 1, max_thresholds)
            thresholds = np.unique(np.quantile(score, qs))

        best = 0.0
        for threshold in thresholds:
            y_pred = (score >= threshold).astype(int)
            best = max(best, affiliation_f1(y_pred))
        return best

    def metric_vus_pr(
        self,
        label: Iterable[int] | np.ndarray,
        score: Iterable[float] | np.ndarray,
        sliding_window: int = 100,
        version: str = "opt",
        thre: int = 250,
    ) -> float:
        """Compute VUS-PR using the extracted standalone implementation."""
        labels_arr = np.asarray(label).astype(int)
        score_arr = np.asarray(score, dtype=float)

        if version == "opt_mem":
            _, _, _, _, _, vus_pr = self.RangeAUC_volume_opt_mem(
                labels_original=labels_arr,
                score=score_arr,
                windowSize=sliding_window,
                thre=thre,
            )
        else:
            _, _, _, _, _, vus_pr = self.RangeAUC_volume_opt(
                labels_original=labels_arr,
                score=score_arr,
                windowSize=sliding_window,
                thre=thre,
            )

        return float(vus_pr)

    def metric_vus_roc(
        self,
        label: Iterable[int] | np.ndarray,
        score: Iterable[float] | np.ndarray,
        sliding_window: int = 100,
        version: str = "opt",
        thre: int = 250,
    ) -> float:
        """Compute VUS-ROC using the extracted standalone implementation."""
        labels_arr = np.asarray(label).astype(int)
        score_arr = np.asarray(score, dtype=float)

        if version == "opt_mem":
            _, _, _, _, vus_roc, _ = self.RangeAUC_volume_opt_mem(
                labels_original=labels_arr,
                score=score_arr,
                windowSize=sliding_window,
                thre=thre,
            )
        else:
            _, _, _, _, vus_roc, _ = self.RangeAUC_volume_opt(
                labels_original=labels_arr,
                score=score_arr,
                windowSize=sliding_window,
                thre=thre,
            )

        return float(vus_roc)

    def __call__(self, metric_name: str, **kwargs) -> float:
        if metric_name not in self.METRIC_MAP:
            raise ValueError(f"Unsupported metric: {metric_name}")

        spec = self.METRIC_MAP[metric_name]
        allowed = set(spec["requires"]) | set(spec.get("params", {}))
        unknown = set(kwargs) - allowed
        if unknown:
            raise ValueError(
                f"Unsupported parameters for {metric_name}: {sorted(unknown)}"
            )

        missing = [
            name for name in spec["requires"]
            if name not in kwargs
        ]
        if missing:
            raise ValueError(
                f"{metric_name} requires inputs: {missing}"
            )

        method = getattr(self, spec["method"])
        inputs = {
            name: kwargs[name]
            for name in spec["requires"]
        }
        params = spec.get("params", {}).copy()
        for name in params:
            if name in kwargs:
                params[name] = kwargs[name]

        return float(method(**inputs, **params))
    

def concordant_count_and_rate(val_metrics, test_metrics):
    v, f = val_metrics, test_metrics
    dv = v[:, None] - v[None, :]
    df = f[:, None] - f[None, :]
    m = np.triu(np.ones((len(v), len(v)), dtype=bool), 1)  # i<j
    C = int(np.sum((dv * df > 0) & m))
    total = len(v) * (len(v) - 1) // 2
    return C, (C / total if total else 0.0)

        
