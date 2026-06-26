"""동작구(11590) backfill 표제부 수집 — 캐시·retry·한도. _experiments 전용, redev/·git 미변경."""
import sys, io, time, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import pandas as pd, requests, urllib3
urllib3.disable_warnings()

KEY = "iCfWPf7ZTlHDbw+k4Kj3+pn2qS4P6B33zISzhgpLu4yvrI0tG2Y/ruww0HrYC4che6+CTEnZlKT/hVlH0/Dv0Q=="
URL = "http://apis.data.go.kr/1613000/BldRgstHubService/getBrTitleInfo"
CACHE = "_experiments/gis_swap/backfill_dongjak.parquet"
DONE = "_experiments/gis_swap/backfill_dongjak_done.csv"   # 처리 완료 PNU(빈 결과 포함) — 이어받기
CALL_BUDGET = 9500

df = pd.read_csv("_experiments/gis_swap/backfill_pnus.csv", dtype={"pnu": str})
df = df[(df["is_res"] == True) | (df["is_res"] == "True")]
dj = [str(x).zfill(19) for x in df[df["gu"].astype(str) == "11590"]["pnu"]]
done = set(pd.read_csv(DONE, dtype={"pnu": str})["pnu"]) if os.path.exists(DONE) else set()
todo = [p for p in dj if p not in done]
print(f"동작 backfill 대상 {len(dj)} · 기수집 {len(done)} · 이번 todo {len(todo)}")

def flush(rows, done_new):
    if rows:
        old = pd.read_parquet(CACHE) if os.path.exists(CACHE) else pd.DataFrame()
        pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_parquet(CACHE, index=False)
    prev = set(pd.read_csv(DONE, dtype={"pnu": str})["pnu"]) if os.path.exists(DONE) else set()
    pd.DataFrame({"pnu": sorted(prev | set(done_new))}).to_csv(DONE, index=False)

rows, done_new, calls = [], [], 0
sess = requests.Session()
for i, p in enumerate(todo):
    if calls >= CALL_BUDGET:
        print(f"★한도 근접({calls}) — 중단. 나머지 {len(todo)-i}건 익일."); break
    sgg, bjd, plat, bun, ji = p[:5], p[5:10], ("0" if p[10] == "1" else "1"), p[11:15], p[15:19]
    params = {"serviceKey": requests.utils.unquote(KEY), "sigunguCd": sgg, "bjdongCd": bjd,
              "platGbCd": plat, "bun": bun, "ji": ji, "numOfRows": 99999, "pageNo": 1, "_type": "json"}
    ok = False
    for attempt in range(3):
        try:
            calls += 1
            r = sess.get(URL, params=params, verify=False, timeout=40)
            j = r.json()
            it = j["response"]["body"]["items"]
            items = [] if not it else (it["item"] if isinstance(it["item"], list) else [it["item"]])
            for x in items:
                rows.append({"pnu": p, "approval_year": str(x.get("useAprDay", ""))[:4],
                             "gross_floor_area": x.get("totArea"), "structure": x.get("strctCdNm"),
                             "dong": (x.get("dongNm") or x.get("bldNm") or "").strip()})
            ok = True
            break
        except Exception:
            time.sleep(0.5 * (attempt + 1))    # backoff
    done_new.append(p)
    time.sleep(0.3)
    if (i + 1) % 300 == 0:                      # ★주기적 flush(받는 즉시 보존·이어받기)
        flush(rows, done_new); rows, done_new = [], []
        print(f"  진행 {i+1}/{len(todo)} · 호출 {calls} · flush 완료")

flush(rows, done_new)                            # 마지막 flush
tot = pd.read_parquet(CACHE) if os.path.exists(CACHE) else pd.DataFrame()
filled = tot[tot["approval_year"].astype(str).str.len() == 4]["pnu"].nunique() if len(tot) else 0
alldone = set(pd.read_csv(DONE, dtype={"pnu": str})["pnu"])
print(f"\n=== 동작 수집 현황 ===")
print(f"누적 처리 PNU {len(alldone)}/{len(dj)} · 이번 호출 {calls}")
print(f"누적 캐시 건물행 {len(tot)} · 사용승인일 채워진 PNU {filled} ({filled/max(len(alldone),1)*100:.0f}% of 처리)")
print("완료" if len(alldone) >= len(dj) else f"미완 — 나머지 {len(dj)-len(alldone)}건 (한도/중단)")
