"""train.py — 심장1 학습 (per-t ego-graph GraphSAGE, R7·R8·R9). 설계: gnn.md.

핵심(gnn.md §2): 라벨노드 v(t_v)의 2홉 ego를 t_v 스냅샷에서 forward → 임베딩. t를
가로질러 concat 후 단일 loss → 한 배치에 양성(옛 t)·음성(2026) 공존(클래스 부재 회피).
구·t별 스냅샷은 한 번 precompute·캐시(구간 엣지=0이라 구별 분해, LODO와 정합).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from redev.config import load_graph_config
from redev.graph.features import FEATURE_COLUMNS, node_features
from redev.models.baseline import _closure, _csr_neighbors
from redev.models.gnn.model import RedevSAGE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # ★자동감지(cuda 가정 금지)
_AREA_IDX = FEATURE_COLUMNS.index("area_m2")                            # log1p 대상


class Normalizer:
    """area_m2 log1p + 전체 z-score. ★통계량은 train fold에서만 적합(누수 차단, gnn.md §5)."""

    def __init__(self):
        self.mean = self.std = None

    def fit(self, feats: np.ndarray) -> "Normalizer":
        f = feats.astype(np.float64).copy()
        f[:, _AREA_IDX] = np.log1p(f[:, _AREA_IDX])     # 555K㎡ outlier 압축
        self.mean, self.std = f.mean(0), f.std(0) + 1e-6
        return self

    def transform(self, feats: np.ndarray) -> np.ndarray:
        f = feats.astype(np.float64).copy()
        f[:, _AREA_IDX] = np.log1p(f[:, _AREA_IDX])
        return ((f - self.mean) / self.std).astype(np.float32)


def build_district_tgroups(aug, edge_index, pnu_to_idx, parcels, buildings, *, hops=2) -> dict:
    """(자치구 d, 시점 t)별 ego 스냅샷 묶음 — 학습/예측 공용, 한 번 계산.

    각 묶음: {t, feat[Nc,5] as-of t, edges[2,El] 로컬, rows(라벨행 idx), local(폐포 내 위치)}.
    구간 엣지=0이라 구별로 분해돼 LODO에서 train구만 골라 forward 가능.
    """
    src_s, dst_s = _csr_neighbors(edge_index, len(pnu_to_idx))
    n_nodes = len(pnu_to_idx)
    idx2pnu = np.empty(n_nodes, dtype=object)
    for p, i in pnu_to_idx.items():
        idx2pnu[i] = p
    lab_global = aug["pnu"].map(pnu_to_idx).to_numpy()
    sig = aug["sigungu"].to_numpy()
    tt = aug["t"].to_numpy()

    groups: dict = {}
    for d in sorted(set(sig)):
        glist = []
        for t in sorted(set(tt[sig == d])):
            rows = np.where((sig == d) & (tt == t))[0]
            V = np.unique(lab_global[rows])
            clo = _closure(V, src_s, dst_s, hops)
            g2l = -np.ones(n_nodes, dtype=np.int64)
            g2l[clo] = np.arange(clo.size)
            m = (g2l[src_s] >= 0) & (g2l[dst_s] >= 0)
            edges = np.stack([g2l[src_s[m]], g2l[dst_s[m]]])
            feat = node_features(
                pd.DataFrame({"pnu": idx2pnu[clo], "t": int(t)}), parcels, buildings,
            )[FEATURE_COLUMNS].to_numpy(np.float32)
            glist.append({"t": int(t), "feat": feat, "edges": edges,
                          "rows": rows, "local": g2l[lab_global[rows]]})
        groups[d] = glist
    return groups


def _forward_collect(model, tgroups, normalizer, *, train_mode):
    """t-그룹들을 forward → (rows, logits) 누적. 한 번에 모아 클래스 혼합 loss(gnn.md §2)."""
    model.train(train_mode)
    rows_all, logits_all = [], []
    for g in tgroups:
        x = torch.from_numpy(normalizer.transform(g["feat"])).to(DEVICE)
        ei = torch.from_numpy(g["edges"]).long().to(DEVICE)
        out = model(x, ei)                              # [Nc] logit
        logits_all.append(out[torch.from_numpy(g["local"]).long().to(DEVICE)])
        rows_all.append(g["rows"])
    return np.concatenate(rows_all), torch.cat(logits_all)


def fit_gnn(train_tgroups, y, loss_rows, val_rows, *, params, pretrained=None, max_epochs=200, patience=15):
    """파인튜닝 — t-그룹 concat 단일 loss, inner-val PR-AUC로 early stopping(CPU).

    loss_rows=손실 계산 노드(버퍼 적용된 train), val_rows=early stopping val. 정규화는
    loss_rows self-피처에서만 적합(누수 차단). 클래스 혼합 카운트 출력(수검 §9-2).
    """
    from redev.eval.metrics import pr_auc

    # loss노드 self피처(정규화 적합용): 라벨 self = feat[local], 그 중 loss_rows만
    tr_local = [g["feat"][g["local"]][np.isin(g["rows"], loss_rows)] for g in train_tgroups]
    norm = Normalizer().fit(np.concatenate([f for f in tr_local if len(f)]))

    model = RedevSAGE(len(FEATURE_COLUMNS), hidden=params["hidden"], dropout=params["dropout"]).to(DEVICE)
    if pretrained is not None:
        model.load_state_dict(pretrained, strict=False)        # 사전학습 가중치 초기화(R7)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["wd"])

    rows0, _ = _forward_collect(model, train_tgroups, norm, train_mode=False)
    is_loss = np.isin(rows0, loss_rows)
    is_val = np.isin(rows0, val_rows)
    y0 = y[rows0]
    pos_w = torch.tensor([(y0[is_loss] == 0).sum() / max(1, (y0[is_loss] == 1).sum())],
                         dtype=torch.float32, device=DEVICE)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    cls = {"loss_pos": int((y0[is_loss] == 1).sum()), "loss_neg": int((y0[is_loss] == 0).sum()),
           "val_pos": int((y0[is_val] == 1).sum()), "val_neg": int((y0[is_val] == 0).sum())}

    best, best_state, bad, ep_times = -1.0, None, 0, []
    ep = 0
    for ep in range(max_epochs):
        t0 = time.time()
        rows, logits = _forward_collect(model, train_tgroups, norm, train_mode=True)
        yv = torch.from_numpy(y[rows].astype(np.float32)).to(DEVICE)
        mtr = torch.from_numpy(np.isin(rows, loss_rows)).to(DEVICE)
        opt.zero_grad()
        lossf(logits[mtr], yv[mtr]).backward()
        opt.step()
        ep_times.append(time.time() - t0)
        with torch.no_grad():
            rows_v, logits_v = _forward_collect(model, train_tgroups, norm, train_mode=False)
            mva = np.isin(rows_v, val_rows)
            auc = pr_auc(y[rows_v][mva], torch.sigmoid(logits_v).cpu().numpy()[mva])
        if auc > best:
            best, best_state, bad = auc, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, norm, {"best_val_prauc": best, "epochs": ep + 1,
                         "sec_per_epoch": float(np.mean(ep_times)), "class_mix": cls}


def predict_gnn(model, norm, test_tgroups, n_rows) -> np.ndarray:
    """test 구 t-그룹 forward → 행별 양성확률."""
    p = np.full(n_rows, np.nan)
    with torch.no_grad():
        rows, logits = _forward_collect(model, test_tgroups, norm, train_mode=False)
        p[rows] = torch.sigmoid(logits).cpu().numpy()
    return p


def _inner_val_and_loss(train_rows, aug, edge_index, pnu_to_idx, *, hops):
    """train 안 단일 공간 holdout(val=zone그룹0) + ★val 2홉 버퍼를 loss에서 제외(gnn.md §5).

    inner-val이 loss노드와 수용영역을 공유하면 early stopping 신호가 부푼다 → 버퍼로 차단.
    """
    from redev.eval.spatial_cv import _hop_neighbors, spatial_zone_groups
    g = spatial_zone_groups(train_rows, aug, k=2)
    val_rows = g[0]
    g_idx = aug["pnu"].map(pnu_to_idx).to_numpy()
    nbr = _hop_neighbors(g_idx[val_rows], edge_index, hops)        # val의 2홉(전역)
    buffered = np.isin(g_idx, np.fromiter(nbr, np.int64) if nbr else np.empty(0, np.int64))
    val_set = set(val_rows.tolist())
    loss_rows = np.array([r for r in train_rows if r not in val_set and not buffered[r]], dtype=np.int64)
    return loss_rows, val_rows


_GNN_COMBOS = [{"hidden": h, "dropout": d, "lr": 0.01, "wd": 5e-4}
               for h in (32, 64) for d in (0.3, 0.5)]


def run_gnn_cv(aug, edge_index, pnu_to_idx, parcels, buildings, *,
               cfg=None, tgroups=None, pretrained=None, pretrain=False,
               fixed_params=None, model_name="GNN") -> dict:
    """GNN 전체 평가: 하파선택(단일 공간holdout, 한세트 고정) → LODO 채점(run_xgb_cv 대응).

    ★베이스라인과 같은 spatial_cv.evaluate·같은 fold → 점수표 한 줄(R9 공정). per-fold·
    per-epoch wall-clock·클래스혼합을 리포트에 남김(환경요청·수검).
    """
    from redev.eval.spatial_cv import build_lodo_folds, evaluate
    cfg = cfg or load_graph_config()
    hops = cfg["cv"]["buffer_hops"]
    G = tgroups or build_district_tgroups(aug, edge_index, pnu_to_idx, parcels, buildings, hops=hops)
    y = aug["y"].to_numpy()
    sig = aug["sigungu"].to_numpy()
    folds = build_lodo_folds(aug, edge_index, pnu_to_idx, cfg=cfg)

    # 하파선택: 전체에서 단일 공간 holdout(한세트 고정, fold별 튜닝=과적합).
    # fixed_params 주어지면 선택 생략(이미 고정된 combo 재사용 — 예측 재생성·재현용).
    if fixed_params is not None:
        best, sel = fixed_params, {"fixed": fixed_params}
    else:
        all_tg = [g for d in G for g in G[d]]
        lr_sel, vr_sel = _inner_val_and_loss(np.arange(len(aug)), aug, edge_index, pnu_to_idx, hops=hops)
        sel = {}
        for c in _GNN_COMBOS:
            _, _, h = fit_gnn(all_tg, y, lr_sel, vr_sel, params=c, pretrained=pretrained)
            sel[f"h{c['hidden']}_d{c['dropout']}"] = round(h["best_val_prauc"], 4)
        best = max(_GNN_COMBOS, key=lambda c: sel[f"h{c['hidden']}_d{c['dropout']}"])

    fold_hist = {}

    def predict_fn(train_idx, test_idx):
        tr_tg = [g for d in set(sig[train_idx]) for g in G[d]]
        te_tg = [g for d in set(sig[test_idx]) for g in G[d]]
        lr, vr = _inner_val_and_loss(train_idx, aug, edge_index, pnu_to_idx, hops=hops)
        init = pretrained
        if pretrain:                                   # ★per-fold 자기지도(통계량 fold-train만)
            from redev.models.gnn.pretrain import pretrain_masked
            tr_feats = np.concatenate([g["feat"][g["local"]][np.isin(g["rows"], train_idx)] for g in tr_tg])
            init = pretrain_masked(tr_tg, Normalizer().fit(tr_feats), hidden=best["hidden"], dropout=best["dropout"])
        model, norm, h = fit_gnn(tr_tg, y, lr, vr, params=best, pretrained=init)
        fold_hist[sorted(set(sig[test_idx]))[0]] = h
        return predict_gnn(model, norm, te_tg, len(aug))[test_idx]

    rep = evaluate(predict_fn, folds, aug, model=model_name)
    rep["selection"] = {"combos_val_prauc": sel, "chosen": f"hidden={best['hidden']},dropout={best['dropout']}"}
    rep["fold_history"] = {k: {"epochs": v["epochs"], "sec_per_epoch": round(v["sec_per_epoch"], 2),
                               "class_mix": v["class_mix"]} for k, v in fold_hist.items()}
    return rep
