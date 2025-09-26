#!/usr/bin/env python3
"""
method2_like_method1.py

Method2 features (GLCM then LBP then intensity stats) but structured exactly like Method1.
Outputs:
 - features.pkl   (X, y, groups, cfg)
 - report.json    (summary + CV metrics + permutation p-value)
 - sample_rois.png
 - final_model.pkl

Usage:
    python method2_like_method1.py --images all-mias --meta data2.txt --outdir results_method2
"""

import os
import argparse
import time
import math
import json
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.base import clone

# ---------------------------
# Helper functions
# ---------------------------
def read_metadata(meta_path):
    col_names = ['REFNUM', 'BG', 'CLASS', 'SEVERITY', 'X', 'Y', 'RADIUS']
    # try header/no-header robust reading
    try:
        df = pd.read_csv(meta_path, sep=r"\s+", names=col_names, header=0)
    except Exception:
        df = pd.read_csv(meta_path, sep=r"\s+", names=col_names, header=None)
    # Convert RADIUS to numeric (coerce invalid -> NaN)
    df['RADIUS'] = pd.to_numeric(df['RADIUS'], errors='coerce')
    df['CANCER'] = df['SEVERITY'].apply(lambda x: 1 if str(x).upper() in ['B','M'] else 0)
    # patient id (mdb001/mdb002 -> patient 1, etc.)
    def ref_to_patient(ref):
        s = str(ref)
        digits = ''.join([c for c in s if c.isdigit()])
        if digits == '':
            return None
        n = int(digits)
        return ((n - 1) // 2) + 1
    df['patient_id'] = df['REFNUM'].map(ref_to_patient)
    return df

def preprocess_img(img):
    """CLAHE for contrast equalization; expects grayscale uint8."""
    if img is None:
        return None
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def roi_from_row(img_eq, row, image_size=1024, median_radius=48, min_side=32):
    """
    Return a square ROI (numpy array). If labeled lesion exists use (X,Y,RADIUS) else center crop with median_radius.
    Pads if too small using reflect border.
    """
    label = int(row['CANCER'])
    x = row['X']; y = row['Y']; r = row['RADIUS']
    H, W = img_eq.shape
    if label == 1 and pd.notna(x) and pd.notna(y) and pd.notna(r):
        cx = int(x)
        cy = int(image_size - float(y))  # MIAS bottom-left -> top-left
        rad = int(r)
        x0 = max(0, cx - rad); x1 = min(W, cx + rad)
        y0 = max(0, cy - rad); y1 = min(H, cy + rad)
        roi = img_eq[y0:y1, x0:x1]
        if roi.size == 0:
            roi = img_eq.copy()
    else:
        rad = int(median_radius)
        cx, cy = W//2, H//2
        x0 = max(0, cx - rad); x1 = min(W, cx + rad)
        y0 = max(0, cy - rad); y1 = min(H, cy + rad)
        roi = img_eq[y0:y1, x0:x1]

    h, w = roi.shape
    if h < min_side or w < min_side:
        top = max(0, (min_side - h) // 2); bottom = max(0, min_side - h - top)
        left = max(0, (min_side - w) // 2); right = max(0, min_side - w - left)
        roi = cv2.copyMakeBorder(roi, top, bottom, left, right, cv2.BORDER_REFLECT)
    return roi

# --- Feature extraction: GLCM first (as requested), then LBP, then intensity stats
def compute_glcm_features(roi, distances=(1,), angles=(0, np.pi/4, np.pi/2, 3*np.pi/4), levels=32):
    """
    Quantize ROI to `levels`, compute GLCM and return a small vector of properties (mean across distances & angles).
    """
    if levels < 2:
        raise ValueError("levels must be >= 2")
    roi_q = (roi / (256.0 / levels)).astype(np.uint8)
    glcm = graycomatrix(roi_q, distances=list(distances), angles=list(angles), levels=levels, symmetric=True, normed=True)
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    feats = []
    for p in props:
        vals = graycoprops(glcm, p)
        feats.append(float(vals.mean()))
    return np.array(feats, dtype=float)

def compute_lbp_hist(roi, P=8, radii=(1,3), n_bins=59):
    """
    For each radius compute uniform LBP and 59-bin histogram (P=8 uniform -> 59 bins).
    Concatenate histograms and L1-normalize per-region (here we normalize per-hist).
    """
    feats = []
    for R in radii:
        lbp = local_binary_pattern(roi, P, R, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        s = hist.sum()
        if s == 0:
            feats.extend([0.0]*n_bins)
        else:
            feats.extend((hist.astype(float) / s).tolist())
    return np.array(feats, dtype=float)

def intensity_stats(roi):
    return np.array([float(roi.mean()), float(roi.std()), float(roi.min()), float(roi.max())], dtype=float)

def convert(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)

# Group-wise permutation test
def group_permutation_test(pipeline, X_train, y_train, groups_train, X_test, y_test, n_permutations=200, random_state=42):
    """
    Permute labels at the group (patient) level on the training set only, refit pipeline on permuted labels,
    and evaluate on the fixed test set. Returns (real_auc, perm_scores_array, pvalue).
    """
    rng = np.random.RandomState(random_state)
    unique_groups = np.unique(groups_train)
    # compute majority label per group in training set
    group_labels = {}
    for g in unique_groups:
        idxs = np.where(groups_train == g)[0]
        # majority label for this group's samples
        group_labels[g] = int(pd.Series(y_train[idxs]).mode().iloc[0])
    group_list = list(unique_groups)

    # fit pipeline on original train to get real AUC
    pipeline.fit(X_train, y_train)
    if hasattr(pipeline, "predict_proba"):
        real_prob = pipeline.predict_proba(X_test)[:,1]
    else:
        real_prob = pipeline.decision_function(X_test)
    try:
        real_auc = float(roc_auc_score(y_test, real_prob))
    except Exception:
        real_auc = float('nan')

    perm_scores = []
    for i in range(n_permutations):
        permuted_group_labels = rng.permutation([group_labels[g] for g in group_list])
        y_train_perm = y_train.copy()
        for g_val, perm_lab in zip(group_list, permuted_group_labels):
            idxs = np.where(groups_train == g_val)[0]
            y_train_perm[idxs] = perm_lab
        # clone pipeline and fit on permuted labels
        pipe_clone = clone(pipeline)
        pipe_clone.fit(X_train, y_train_perm)
        if hasattr(pipe_clone, "predict_proba"):
            prob_p = pipe_clone.predict_proba(X_test)[:,1]
        else:
            prob_p = pipe_clone.decision_function(X_test)
        try:
            auc_p = float(roc_auc_score(y_test, prob_p))
        except Exception:
            auc_p = float('nan')
        perm_scores.append(auc_p)
    perm_arr = np.array(perm_scores, dtype=float)
    # p-value (plus-one rule)
    pvalue = (np.sum(perm_arr >= real_auc) + 1) / (len(perm_arr) + 1)
    return real_auc, perm_arr, pvalue

# ---------------------------
# Main pipeline (Method1 structure, Method2 logic)
# ---------------------------
def main(args):
    start_time = time.time()
    cfg = {
        'P': args.P,
        'lbp_radii': tuple(args.lbp_radii),
        'lbp_bins': args.lbp_bins,
        'glcm_distances': tuple(args.glcm_distances),
        'glcm_angles': (0, math.pi/4, math.pi/2, 3*math.pi/4),
        'glcm_levels': args.glcm_levels,
    }

    # read metadata
    df = read_metadata(args.meta)
    print(f"[INFO] metadata rows: {len(df)}")
    radii_num = pd.to_numeric(df['RADIUS'], errors='coerce').dropna()
    median_radius = int(radii_num.median()) if radii_num.size > 0 else args.median_radius
    print(f"[INFO] median radius: {median_radius}")

    # list images
    images = sorted([f for f in os.listdir(args.images) if f.lower().endswith('.pgm')])
    print(f"[INFO] found {len(images)} image files in {args.images}")

    features = []
    labels = []
    groups = []
    sample_rois = []

    for fname in images:
        ref = os.path.splitext(fname)[0]
        row = df[df['REFNUM'] == ref]
        if row.empty:
            continue
        row = row.iloc[0]
        img_path = os.path.join(args.images, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] cannot read {img_path}, skipping")
            continue
        try:
            img_eq = preprocess_img(img)
            roi = roi_from_row(img_eq, row, image_size=args.image_size, median_radius=median_radius, min_side=args.min_side)
            # resize to target_size (keep consistent)
            roi_resized = cv2.resize(roi, (args.target_size, args.target_size), interpolation=cv2.INTER_AREA)
            # Method2 logic but GLCM first then LBP
            glcm_feats = compute_glcm_features(roi_resized, distances=cfg['glcm_distances'], angles=cfg['glcm_angles'], levels=cfg['glcm_levels'])
            lbp_feats = compute_lbp_hist(roi_resized, P=cfg['P'], radii=cfg['lbp_radii'], n_bins=cfg['lbp_bins'])
            int_feats = intensity_stats(roi_resized)
            feat_vec = np.concatenate([glcm_feats, lbp_feats, int_feats]).astype(float)
            features.append(feat_vec)
            labels.append(int(row['CANCER']))
            groups.append(row['patient_id'])
            if len(sample_rois) < 6:
                sample_rois.append((ref, int(row['CANCER']), row['patient_id'], roi_resized))
        except Exception as e:
            print(f"[ERROR] processing {ref}: {e}")

    X = np.vstack(features) if len(features) > 0 else np.zeros((0, cfg['lbp_bins']*len(cfg['lbp_radii']) + 5))
    y = np.array(labels, dtype=int)
    groups_arr = np.array(groups)
    print("[INFO] extracted features shape:", X.shape)
    print(" label distribution:", Counter(y))
    print(" unique groups:", len(set([g for g in groups_arr if g is not None])))

    # save features.pkl
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "features.pkl"), "wb") as f:
        pickle.dump({"X": X, "y": y, "groups": groups_arr, "cfg": cfg}, f)
    print("[INFO] saved features.pkl")

    # save sample rois image
    fig = plt.figure(figsize=(10,4))
    for idx, (ref, lab, grp, roi_img) in enumerate(sample_rois):
        ax = fig.add_subplot(2,3, idx+1)
        ax.imshow(roi_img, cmap='gray'); ax.set_title(f"{ref} L={lab} G={grp}"); ax.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "sample_rois.png"))
    plt.close(fig)
    print("[INFO] saved sample_rois.png")

    # -------------------------
    # Nested group-aware CV
    # -------------------------
    unique_groups = len(set([g for g in groups_arr if g is not None]))
    outer_splits = min(5, max(2, unique_groups))
    inner_splits = max(2, outer_splits - 1)
    print(f"[INFO] using outer_splits={outer_splits}, inner_splits={inner_splits}")

    outer_cv = GroupKFold(n_splits=outer_splits)
    inner_cv = GroupKFold(n_splits=inner_splits)

    # build pipelines (scaler + selector + clf) to avoid leakage
    selector = SelectKBest(mutual_info_classif, k=min(args.select_k, X.shape[1]))
    scaler = StandardScaler()

    pipelines = {
        'rf': Pipeline([('scaler', scaler), ('select', selector), ('clf', RandomForestClassifier(class_weight='balanced', random_state=args.random_state))]),
        'svm': Pipeline([('scaler', scaler), ('select', selector), ('clf', SVC(probability=True, class_weight='balanced', random_state=args.random_state))]),
    }

    # small parameter grids (kept simple — expand if desired)
    rf_grid = {'clf__n_estimators': [args.rf_estimators], 'clf__max_depth': [None], 'clf__min_samples_leaf': [1]}
    svm_grid = {'clf__C': [1.0], 'clf__gamma': ['scale']}

    results = {k: [] for k in pipelines.keys()}
    best_models = {}

    fold = 0
    for train_idx, test_idx in outer_cv.split(X, y, groups_arr):
        fold += 1
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        gtr, gte = groups_arr[train_idx], groups_arr[test_idx]
        print(f"[INFO] Outer fold {fold}: train={len(train_idx)} test={len(test_idx)} groups_train={len(set(gtr))}")

        for name, pipe in pipelines.items():
            param_grid = rf_grid if name == 'rf' else svm_grid
            gs = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring='roc_auc', n_jobs=args.n_jobs, refit=True)
            # IMPORTANT: pass groups so inner GroupKFold uses them
            gs.fit(Xtr, ytr, groups=gtr)
            best = gs.best_estimator_
            # evaluate on outer test
            ypred = best.predict(Xte)
            if hasattr(best, "predict_proba"):
                yprob = best.predict_proba(Xte)[:,1]
            else:
                yprob = best.decision_function(Xte)
            acc = accuracy_score(yte, ypred)
            rec = recall_score(yte, ypred, zero_division=0)
            auc = roc_auc_score(yte, yprob) if len(np.unique(yte)) > 1 else float('nan')
            cm = confusion_matrix(yte, ypred)
            results[name].append({'fold': fold, 'acc': acc, 'recall': rec, 'auc': auc, 'cm': cm, 'best_params': gs.best_params_})
            best_models[name] = gs.best_estimator_
            print(f"[FOLD {fold}][{name}] acc={acc:.3f} rec={rec:.3f} auc={auc:.3f}")

    # summarize
    summary = {
        'n_samples': int(X.shape[0]),
        'n_features': int(X.shape[1]),
        'label_counts': dict(pd.Series(y).value_counts()),
        'unique_groups': int(len(set(groups_arr))),
        'cv': {}
    }
    for name in results:
        arr_acc = np.array([r['acc'] for r in results[name]])
        arr_rec = np.array([r['recall'] for r in results[name]])
        arr_auc = np.array([r['auc'] for r in results[name] if not math.isnan(r['auc'])])
        summary['cv'][name] = {
            'mean_acc': float(np.nanmean(arr_acc)),
            'std_acc': float(np.nanstd(arr_acc)),
            'mean_rec': float(np.nanmean(arr_rec)),
            'std_rec': float(np.nanstd(arr_rec)),
            'mean_auc': float(np.nanmean(arr_auc)) if arr_auc.size>0 else None,
            'std_auc': float(np.nanstd(arr_auc)) if arr_auc.size>0 else None,
            'per_fold': results[name]
        }

    # refit final models on full dataset and save
    final_models = {}
    for name, model in best_models.items():
        try:
            model.fit(X, y)
            final_models[name] = model
            with open(os.path.join(args.outdir, f"final_model_{name}.pkl"), "wb") as f:
                pickle.dump(model, f)
        except Exception as e:
            print(f"[WARN] could not refit/save {name}: {e}")

    # group-wise permutation test on RF (if present)
    if 'rf' in final_models:
        print("[INFO] Running group-wise permutation test for RF...")
        try:
            # use a held-out split for permutation: create a GroupShuffleSplit held-out split
            gss = GroupShuffleSplit(n_splits=1, test_size=args.perm_test_holdout_fraction, random_state=args.random_state)
            train_idx_perm, test_idx_perm = next(gss.split(X, y, groups_arr))
            Xtr_perm, Xte_perm = X[train_idx_perm], X[test_idx_perm]
            ytr_perm, yte_perm = y[train_idx_perm], y[test_idx_perm]
            gtr_perm, gte_perm = groups_arr[train_idx_perm], groups_arr[test_idx_perm]
            real_auc, perm_scores, pvalue = group_permutation_test(final_models['rf'], Xtr_perm, ytr_perm, gtr_perm, Xte_perm, yte_perm, n_permutations=args.n_permutations, random_state=args.random_state)
            summary['perm_test'] = {'real_auc': float(real_auc), 'pvalue': float(pvalue), 'n_permutations': int(args.n_permutations)}
            summary['perm_scores_sample'] = perm_scores[:min(200, len(perm_scores))].tolist()
            print(f"[INFO] Permutation p-value (RF): {pvalue:.4f}")
        except Exception as e:
            print("[WARN] permutation test failed:", e)
    else:
        print("[WARN] RF not available to run permutation test.")

    # save report.json
    with open(os.path.join(args.outdir, "report.json"), "w") as f:
        json.dump(summary, f, indent=2, default=convert)
    print("[INFO] saved report.json")

    elapsed = time.time() - start_time
    print(f"[DONE] elapsed seconds: {elapsed:.1f}")
    print(json.dumps(summary, indent=2, default=convert))

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Method2 (GLCM+LBP) but same structure as Method1")
    parser.add_argument("--images", required=True, help="folder with .pgm images (e.g. all-mias)")
    parser.add_argument("--meta", required=True, help="metadata file (e.g. data2.txt)")
    parser.add_argument("--outdir", default="results_method2", help="folder to save outputs")
    parser.add_argument("--target_size", type=int, default=128, help="resize ROIs to this size")
    parser.add_argument("--min_side", type=int, default=32, help="minimum ROI side (pad if smaller)")
    parser.add_argument("--image_size", type=int, default=1024, help="original image size (MIAS default 1024)")
    parser.add_argument("--P", type=int, default=8, help="LBP P")
    parser.add_argument("--lbp_radii", nargs="+", type=int, default=[1,3], help="LBP radii")
    parser.add_argument("--lbp_bins", type=int, default=59, help="LBP histogram bins (uniform P=8 -> 59)")
    parser.add_argument("--glcm_levels", type=int, default=32, help="GLCM quantization levels")
    parser.add_argument("--glcm_distances", nargs="+", type=int, default=[1], help="GLCM distances")
    parser.add_argument("--select_k", type=int, default=100, help="SelectKBest k")
    parser.add_argument("--rf_estimators", type=int, default=300, help="RF n_estimators (used in grid)")
    parser.add_argument("--n_permutations", type=int, default=200, help="permutation test permutations (group-wise)")
    parser.add_argument("--perm_test_holdout_fraction", type=float, default=0.2, help="fraction for holdout inside permutation test")
    parser.add_argument("--n_jobs", type=int, default=-1, help="n_jobs for GridSearchCV")
    parser.add_argument("--random_state", type=int, default=42, help="random seed")
    args = parser.parse_args()

    # Propagate parsed args we use in main
    args.lbp_radii = list(args.lbp_radii)
    args.glcm_distances = list(args.glcm_distances)
    args.select_k = args.select_k
    args.rf_estimators = args.rf_estimators
    args.perm_test_holdout_fraction = args.perm_test_holdout_fraction
    args.n_permutations = args.n_permutations
    args.n_jobs = args.n_jobs
    args.random_state = args.random_state
    args.target_size = args.target_size
    args.min_side = args.min_side
    args.image_size = args.image_size
    args.outdir = args.outdir

    main(args)
