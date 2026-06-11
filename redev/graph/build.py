"""build.py — 필지 인접 그래프 (심장1 GNN의 입력).

역할: 주거계 필지를 노드로, 경계 공유를 엣지로 하는 그래프를 만든다. 메시지 패싱이
이 그래프 위에서 "이 필지의 *주변 블록*이 재개발 환경인가"를 노드에 흡수한다(graph.md §3-1).

★수정1: 도로·하천 등 비주거 지목은 노드에서 제외(슈퍼허브 방지). ★R10: 구 단위 배치 +
STRtree 공간인덱스(173K 한방 touches는 메모리 폭발). 설계: docs/design/graph.md.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import torch
from shapely import STRtree
from torch_geometric.data import Data

from redev.config import load_graph_config


def node_parcels(
    parcels: gpd.GeoDataFrame,
    building_pnus: set | None = None,
    cfg: dict | None = None,
) -> gpd.GeoDataFrame:
    """비주거 지목(도로·하천·임야 등) 제외 → 노드 필지만 (수정1 + 건물 override).

    왜: 도로 필지가 길게 뻗어 수백 필지와 touches → 슈퍼허브로 메시지가 블록을 넘어
    번진다. 비주거 지목은 노드에서 뺀다(제외 목록 config). ★단 **건물이 있으면 유지**
    (building_overrides_jimok): 측정상 산동네 재개발 구역에 임/전/답 지목의 *주택* 필지가
    섞여 있어(graph.md §3) — 건물 없는 대형 임야만 빠지고 산동네 주택은 산다.
    """
    cfg = cfg or load_graph_config()
    exclude = set(cfg["node_jimok"]["exclude"])
    excluded_jimok = parcels["jimok"].isin(exclude)
    if cfg["node_jimok"].get("building_overrides_jimok") and building_pnus is not None:
        has_building = parcels["pnu"].isin(building_pnus)
        keep = ~excluded_jimok | has_building          # 건물 있으면 비주거여도 유지
    else:
        keep = ~excluded_jimok
    return parcels[keep].copy()


def build_adjacency(nodes: gpd.GeoDataFrame, *, buffer_m: float = 0.0) -> torch.Tensor:
    """경계 공유(또는 buffer_m 내) 노드쌍 → edge_index [2,E] (무향=양방향, 자기루프 0).

    STRtree: 모든 쌍(O(n²)) 대신 공간인덱스로 *근방만* 검사한다(R10). buffer_m>0이면
    도로로 갈린 필지를 잇는다(strict touches는 buffer_m=0).

    반환: torch.LongTensor [2, E] — PyG 표준 포맷. nodes의 행 순서가 곧 노드 인덱스.
    """
    geoms = nodes.geometry.values
    if len(geoms) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    # STRtree = 경계상자 기반 공간인덱스(근방 후보만 추림). predicate로 실제 위상 검사.
    tree = STRtree(geoms)
    if buffer_m > 0:
        query_geoms = np.array([g.buffer(buffer_m) for g in geoms], dtype=object)
        predicate = "intersects"      # buffer 안에 들어오면 이웃(도로 다리)
    else:
        query_geoms = geoms
        predicate = "touches"         # 경계 공유만(도로=블록 경계)
    # 벡터화 질의: [2, M] = (질의필지 i, 트리필지 j) 쌍.
    qi, tj = tree.query(query_geoms, predicate=predicate)
    keep = qi != tj                   # 자기루프 제거
    qi, tj = qi[keep], tj[keep]
    if len(qi) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    # 무향 → (min,max)로 중복 제거 후 양방향 복원.
    a = np.minimum(qi, tj)
    b = np.maximum(qi, tj)
    pairs = np.unique(np.stack([a, b]), axis=1)
    src = torch.as_tensor(np.concatenate([pairs[0], pairs[1]]), dtype=torch.long)
    dst = torch.as_tensor(np.concatenate([pairs[1], pairs[0]]), dtype=torch.long)
    return torch.stack([src, dst])


def build_graph(
    parcels: gpd.GeoDataFrame,
    *,
    building_pnus: set | None = None,
    cfg: dict | None = None,
) -> tuple[Data, dict, gpd.GeoDataFrame]:
    """필지 인접 그래프 (★구 단위 배치, R10) → (PyG Data, pnu→idx, node_parcels).

    구별로 STRtree 인접을 계산해 전역 노드 인덱스로 합친다(구 = 분리 컴포넌트 — 재개발
    구역은 한 자치구 안이라 안전, 구 경계 교차 엣지만 손실. 수검에서 확인).
    building_pnus: 건물 있는 PNU 집합(node_parcels의 건물 override용).

    반환: Data(edge_index, num_nodes), pnu_to_idx(매핑), nodes(피처 계산용 필지).
    """
    cfg = cfg or load_graph_config()
    nodes = node_parcels(parcels, building_pnus, cfg).reset_index(drop=True)
    buffer_m = float(cfg["edge"]["buffer_m"])
    pnu_to_idx = {p: i for i, p in enumerate(nodes["pnu"])}

    edge_parts = []
    for _, grp in nodes.groupby("sigungu", sort=False):
        sub = grp.reset_index(drop=True)
        ei = build_adjacency(sub, buffer_m=buffer_m)          # sub-로컬 인덱스
        if ei.numel():
            local2global = torch.as_tensor(
                [pnu_to_idx[p] for p in sub["pnu"]], dtype=torch.long
            )
            edge_parts.append(local2global[ei])               # 로컬→전역 매핑
    edge_index = torch.cat(edge_parts, dim=1) if edge_parts else torch.empty((2, 0), dtype=torch.long)
    # PyG Data: 그래프 1개를 담는 표준 컨테이너(edge_index·num_nodes·나중에 x/y).
    data = Data(edge_index=edge_index, num_nodes=len(nodes))
    return data, pnu_to_idx, nodes


def reconcile_labels_to_graph(label_table, node_pnus: set) -> tuple:
    """라벨을 그래프 노드 집합으로 한정 (비노드 라벨 drop) — 라벨↔그래프 브리지.

    ★수검 "라벨 PNU 전원 그래프 존재": 학습은 그래프 노드 위에서만 일어나므로 라벨
    PNU는 전원 노드여야 한다. 비노드 라벨(도로·공원 등 인프라 필지에 붙은 것)은 drop.
    측정(graph.md §3): 비노드 positive 2,519 중 도로 2,315 — 구역 폴리곤 안 도로 필지라
    drop이 맞다(도로는 재개발 노드 아님).

    반환: (filtered_table, dropped_report)
    """
    in_graph = label_table["pnu"].isin(node_pnus)
    dropped = label_table[~in_graph]
    report = {
        "kept": int(in_graph.sum()),
        "dropped_non_node": int((~in_graph).sum()),
        "dropped_by_certainty": dropped["certainty"].value_counts().to_dict(),
    }
    return label_table[in_graph].copy(), report
