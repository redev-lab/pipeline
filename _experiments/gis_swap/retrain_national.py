"""#3-c: national 피처로 train_matrix 재생성 + oof PR-AUC 전/후. (학습 소스는 baseline.py서 national 교체됨)"""
import sys, io, time, warnings
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import pandas as pd
from redev.models.baseline import load_training_matrix, prepare_baseline_matrix, production_feature_set
from redev.models.feasibility import oof_scores
from redev.eval.metrics import pr_auc

t0 = time.time()
aug_s = pd.read_parquet("_data/processed/train_matrix_nb.parquet.seoul_bak")
print(f"서울 aug rows {len(aug_s)} pos {int(aug_s['y'].sum())}", flush=True)
tm = load_training_matrix(force_rebuild=True)        # ★national 매트릭스 재생성
aug_n = prepare_baseline_matrix(force_rebuild=True)
print(f"national aug rows {len(aug_n)} pos {int(aug_n['y'].sum())} ({time.time()-t0:.0f}s)", flush=True)
fc = production_feature_set(aug_n)
ps = pr_auc(aug_s["y"], oof_scores(aug_s, tm.edge_index, tm.pnu_to_idx, feat_cols=[c for c in fc if c in aug_s.columns]))
pn = pr_auc(aug_n["y"], oof_scores(aug_n, tm.edge_index, tm.pnu_to_idx, feat_cols=fc))
print(f"PRAUC_SEOUL={ps:.3f} PRAUC_NATIONAL={pn:.3f} VERDICT={'OK' if pn >= ps - 0.03 else 'DROP'} ({time.time()-t0:.0f}s)", flush=True)
print("RETRAIN_DONE", flush=True)
