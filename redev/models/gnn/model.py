"""model.py — RedevSAGE: GraphSAGE 2층 inductive 노드 분류기 (심장1 본체).

역할: "이 필지의 *주변 블록*이 재개발 환경인가"를 학습된 메시지 패싱으로 푼다. 입력은
self 피처 5차원뿐 — 이웃집계는 GNN이 *학습*한다(R9: B1+와 같은 raw 신호·같은 2홉 수용
영역, 집계 방식만 손피처 vs 학습가중). 설계: docs/design/gnn.md §3.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class RedevSAGE(nn.Module):
    """GraphSAGE 2층(얕게) — R7: 유효표본 42구역 대비 과적합 방지.

    왜 2층인가: 수용영역 2홉 = B1+(2홉 손집계)와 동일 → R9 공정 비교(구조 vs 손피처).
    더 깊으면 oversmoothing(이웃평균 반복으로 노드가 구분 안 됨) + 라벨 부족 과적합.
    왜 mean aggregator: neighbor_aggregate의 이웃 mean과 같은 연산을 *학습된 선형변환·
    비선형과 함께* 수행 — "GNN이 손집계 위에 얹는 게 정확히 학습가중/비선형"이 R9 질문.
    """

    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.5):
        super().__init__()
        # SAGEConv: 각 노드 = W1·자기 + W2·(이웃 mean) 후 비선형. '이웃 흡수'의 학습판.
        self.conv1 = SAGEConv(in_dim, hidden, aggr="mean")
        self.conv2 = SAGEConv(hidden, hidden, aggr="mean")
        self.head = nn.Linear(hidden, 1)        # 임베딩 → 양성 logit
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """노드 피처 x[N,in_dim] + edge_index[2,E] → 노드별 logit[N]."""
        h = self.embed(x, edge_index)
        return self.head(h).squeeze(-1)

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """2홉 임베딩(pretrain·infer 공용). dropout은 training 플래그로 자동 on/off."""
        h = F.relu(self.conv1(x, edge_index))           # 1홉 이웃 흡수
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index))           # 2홉(=B1+ 수용영역)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h
