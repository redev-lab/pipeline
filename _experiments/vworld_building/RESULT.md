# VWorld WFS 건물 데이터 ↔ 우리 GIS건물(AL_D010) 구조 비교 (측정, 2026-06-23)

> 목적: 서울 열린데이터광장 GIS건물(4유형) → 국가표준 1유형 swap 시 노후도·호수밀도 계산이
> 그대로 되는지 **컬럼 매핑 구조** 확인(게이트). 일회용 측정, redev/ 본 파이프라인 미변경.

## 호출 결과
- 인증: `domain=localhost`로 통과(키 자체는 유효, 신청 도메인=localhost 추정). domain 빈값·127.0.0.1도 통과.
- **VWorld WFS 건물 레이어 = `lt_c_spbd`(도로명주소건물) 단 하나.** "GIS건물통합정보"라는 별도 레이어 **없음**.
  (177개 FeatureType 중 건물=lt_c_spbd뿐. 건축물대장/표제부/사용승인/연면적 류 레이어 부재 — 용도지역만 존재.)
- GetFeature: 정릉동 bbox(위도,경도 축순서 = EPSG:4326) → **631개 건물** 수신. geometry=MultiPolygon.

## lt_c_spbd properties (23 컬럼)
`pk, bd_mgt_sn, sido, sigungu, gu, rd_nm, bld_s, bld_e, buld_nm, buld_nm_dc, buld_se_cd,
bul_eng_nm, zip_cd, gro_flo_co(지상층수), und_flo_co(지하층수), buld_no, sig_cd, rn_cd,
emd_cd, pnu, xpos, ypos, poi_chk`

## 서울 AL_D010 ↔ VWorld lt_c_spbd 매핑표

| 계산 용도 | 우리 AL_D010 | VWorld lt_c_spbd | 판정 |
|---|---|---|---|
| **노후도(R1)** | A13 사용승인일 → 연도 | **없음** | ❌ **불가** (준공/승인일 컬럼 자체 부재) |
| **연면적/호수밀도** | A14 연면적 | **없음** (gro_flo_co 층수만) | ❌ **불가** (층수 ≠ 연면적, 환산 부정확) |
| 구조 | A11 구조명 | 없음 | ❌ 불가 |
| **PNU 조인** | A2 (19자리) | pnu (19자리, 예 `1129013300106840020`) | ✅ **가능** |
| 시군구 | A23 | sig_cd / sigungu | ✅ |
| (보너스, 우리 미사용) | — | 지상/지하 층수 | — |

## ★한 줄 결론
**VWorld WFS로는 노후도·호수밀도 swap 불가.** 건물 레이어가 `lt_c_spbd`(도로명주소건물)뿐인데
여기엔 **노후도의 닻인 사용승인일도, 연면적도 없다**(PNU·층수·주소만). PNU 조인은 되지만 핵심
속성이 없어 의미 없음.

## 다음 후보 (이 측정 밖)
1유형 건물 *속성*(사용승인일·연면적)은 VWorld WFS가 아니라 **국토부 건축물대장 API(data.go.kr 표제부)**
또는 **GIS건물통합정보의 data.go.kr 배포본** 라이선스 확인이 진짜 경로. 거기서 PNU(또는 관리번호)로
조인 가능한지 + 사용승인일·연면적 제공 여부를 다음 단계에서 측정.

## 산출물(이 폴더)
- `정릉동_lt_c_spbd_sample.json` — GeoJSON 샘플 3건(raw)
- `feat_latlon.json` — 정릉동 631건 전체
- `properties_columns.txt` — 컬럼 목록
- `caps.xml` / `layers.txt` — GetCapabilities 177 레이어
