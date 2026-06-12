"""AVM 시점분리 검증 — 과거 학습→이후 거래 예측, 구평균 베이스라인 대비(R9 정신).
분석 전용. → _data/processed/_avm_validate.txt
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8")
from redev.config import training_sigungu_codes
from redev.data.ingest.building_gis import load_buildings
from redev.data.ingest.parcels import load_parcels
from redev.models.avm import (AVM_FEATURES, _PYUNG_M2, avm_features, build_target,
                              comparable_newbuild, explain, fit_avm, market_context)
from redev.models.baseline import _SRC, _vsizip

codes = sorted(training_sigungu_codes())
parcels, _ = load_parcels(_vsizip(*_SRC["parcels"]), codes, with_geometry=True)
buildings, _ = load_buildings(_vsizip(*_SRC["buildings"]), with_geometry=False)
trades = pd.read_parquet("_data/processed/_trades_36m.parquet")

ym = pd.to_numeric(trades["deal_ym"], errors="coerce")
SPLIT = 202501
train = trades[ym < SPLIT]
test = trades[(ym >= SPLIT) & (trades["trade_type"] == "villa")
              & trades["land_share_m2"].notna() & (trades["land_share_m2"] > 0)].copy()

# 과거만으로 타깃·피처·모델
tt = build_target(parcels, train, current_ym=str(SPLIT))
feat = avm_features(parcels, buildings, tt)
reliable = feat[feat["agg_level"].isin(["r50", "r100"])].dropna(subset=["target_pyung", *AVM_FEATURES])
model = fit_avm(reliable)

# 이후 거래로 검증
test["actual"] = test["deal_amount"] / (test["land_share_m2"] / _PYUNG_M2)
fmap = feat.set_index("pnu")
tx = test[["pnu", "actual"]].join(fmap[AVM_FEATURES], on="pnu").dropna(subset=AVM_FEATURES)
pred = model.predict(tx[AVM_FEATURES])
a = tx["actual"].to_numpy()
mae = float(np.abs(pred - a).mean())
mape = float((np.abs(pred - a) / a).mean())

# 구평균 베이스라인
gu_med = reliable.assign(gu=reliable["pnu"].str[:5]).groupby("gu")["target_pyung"].median()
bpred = tx["pnu"].str[:5].map(gu_med).to_numpy()
bmae = float(np.abs(bpred - a).mean())
bmape = float((np.abs(bpred - a) / a).mean())

L = []
L.append(f"학습 거래 {len(train)} (deal_ym < {SPLIT}) / 검증 villa 거래 {len(test)} (>= {SPLIT})")
L.append(f"★시점분리 증명: 학습 max_ym {int(ym[ym<SPLIT].max())} < 검증 min_ym {int(pd.to_numeric(test.deal_ym).min())} "
         f"→ {'OK 미래누수 0' if int(ym[ym<SPLIT].max()) < int(pd.to_numeric(test.deal_ym).min()) else '★누수'}")
L.append(f"검증 매칭 필지피처 {len(tx)} / 학습 reliable {len(reliable)}")
L.append("")
L.append(f"★AVM      MAE {mae:.0f} 만원/평 | MAPE {100*mape:.1f}%")
L.append(f"  구평균 베이스라인 MAE {bmae:.0f} | MAPE {100*bmape:.1f}%")
L.append(f"  → AVM이 베이스라인 {'이김' if mae < bmae else '못이김(데이터 병목 보고)'} "
         f"(MAE {bmae-mae:+.0f}, {100*(bmape-mape):+.1f}%p)")
L.append("")
L.append(f"가치 기여: {explain(model, reliable)}")
L.append("")
# 상승여력 밴드 예시(한 필지)
comp = comparable_newbuild(parcels, train)
ex = reliable.merge(comp, on="pnu").dropna(subset=["comp_pyung"]).iloc[0]
mc = market_context(ex["target_pyung"], ex["comp_pyung"], agg_level=ex["agg_level"], n_trades=int(ex["n_trades"]))
L.append(f"시세 맥락 예시 PNU …{ex['pnu'][-6:]}: 빌라 대지지분 {mc['land_share_pyung_man']:.0f} / "
         f"신축 전용 {mc['newbuild_exclu_pyung_man']:.0f} 만원/평 (빼지 않음 — 단위 다름)")
open("_data/processed/_avm_validate.txt", "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
print("DONE")
