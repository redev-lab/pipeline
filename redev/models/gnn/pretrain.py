"""pretrain.py — 자기지도 사전학습 (마스킹 피처 복원, R7). 설계: gnn.md §4.

라벨 ~42구역은 적다. 비라벨 노드(이웃 포함)에서 피처 일부를 가리고 그래프 구조로 복원
하게 해 표현을 먼저 배운다 → 파인튜닝 초기화. ★기대치(gnn.md §4): v1 피처 5차원은
얇아 리프트가 작거나 0일 수 있다 — 그 자체가 "v1 피처 폭에선 사전학습 무용"이라는 측정.
★DGI 아닌 마스킹 복원: "리프트 없으면 버림"이면 싼 것부터. 정규화 통계는 fold-train만.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from redev.graph.features import FEATURE_COLUMNS
from redev.models.gnn.model import RedevSAGE
from redev.models.gnn.train import DEVICE


class _MaskedRecon(nn.Module):
    """RedevSAGE 인코더 + 선형 디코더 — 가린 노드 피처를 이웃으로 복원."""

    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.enc = RedevSAGE(in_dim, hidden, dropout)
        self.dec = nn.Linear(hidden, in_dim)

    def forward(self, x, ei):
        return self.dec(self.enc.embed(x, ei))


def pretrain_masked(tgroups, normalizer, *, hidden, dropout=0.3, lr=0.01,
                    epochs=60, mask_rate=0.3, seed=0) -> dict:
    """t-그룹 폐포(비라벨 포함)에서 마스킹 복원 학습 → 인코더 state_dict 반환.

    가린 노드 피처를 0으로 두고, 이웃 메시지로 원값을 복원(MSE). conv1·conv2 가중치가
    "이웃으로 노드를 설명하는" 표현을 배운다 → fit_gnn이 strict=False로 이식.
    """
    torch.manual_seed(seed)
    net = _MaskedRecon(len(FEATURE_COLUMNS), hidden, dropout).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
    snaps = [(torch.from_numpy(normalizer.transform(g["feat"])).to(DEVICE),
              torch.from_numpy(g["edges"]).long().to(DEVICE)) for g in tgroups]
    net.train()
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for x, ei in snaps:
            mask = torch.rand(x.shape[0], generator=g) < mask_rate
            if not mask.any():
                continue
            xin = x.clone()
            xin[mask] = 0.0                                  # 가림
            out = net(xin, ei)
            loss = ((out[mask] - x[mask]) ** 2).mean()       # 가린 곳만 복원 MSE
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net.enc.state_dict()
