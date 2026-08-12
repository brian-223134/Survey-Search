"""P0.2 / P0.3 — FAISS 인덱스 id 매핑 실측과 검색 지연 측정.

**이 모듈의 목적은 재측정이 아니라 고정입니다.** SETTING.md §6-A 에 적힌
"왕복 10/10 통과"가 코드로 재현되지 않으면 그냥 문서 속 숫자일 뿐입니다.
`tests/test_faiss_mapping.py` 가 이걸 회귀 테스트로 잡습니다.

확인하는 것 세 가지:

1. **id 매핑** — `IndexIDMap.id_map` 이 `[1..N]` 1-based 연속인가,
   `arxivid_to_index_abs.json` 의 값 범위와 일치하는가
2. **의미 수준 왕복** — DB의 논문 → title+abs 임베딩 → 검색 → top-1 id → arxiv_id 가
   원래 논문과 같은가. 이게 진짜 검증입니다. 1번만 맞고 2번이 틀리면
   매핑은 맞는데 임베딩 텍스트 구성이 다른 것입니다.
3. **지연** — 배치 검색의 쿼리당 시간. facet fan-out 설계의 근거

사용:

    python -m survey_search.index.inspect_faiss --probe 10
    python -m survey_search.index.inspect_faiss --probe 10 --skip-semantic  # GPU 없이
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np

from survey_search.assets import EMBED_DIM, FAISS_ID_BASE, GTE_MODEL, N_PAPERS, SURVEYFORGE

log = logging.getLogger(__name__)


@dataclass
class MappingReport:
    """id 매핑 실측 결과. `ok` 가 False면 이후 단계는 전부 신뢰할 수 없습니다."""

    index_type: str
    metric: str
    ntotal: int
    dim: int
    id_min: int
    id_max: int
    id_is_permutation: bool      # id 집합 == {base .. base+N-1} (전단사)
    id_row_aligned: bool         # id_map[row] == row + base — **실측 결과 False**
    n_row_misaligned: int        # 행 순서와 어긋나는 항목 수
    id_base: int                 # 실측된 시작값 (1이어야 정상)
    json_entries: int
    json_min: int
    json_max: int
    json_matches_index: bool     # json 값 집합 == index id 집합
    roundtrip_total: int = 0
    roundtrip_ok: int = 0
    roundtrip_detail: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """`id_row_aligned` 는 **의도적으로 빼 놓았습니다.** 실제로 False이고,
        그래도 검색은 정상입니다 — id로만 접근하는 한 문제가 없기 때문입니다."""
        return (
            self.id_is_permutation
            and self.id_base == FAISS_ID_BASE
            and self.json_matches_index
            and (self.roundtrip_total == 0 or self.roundtrip_ok == self.roundtrip_total)
        )

    def summary(self) -> str:
        rt = (
            f"{self.roundtrip_ok}/{self.roundtrip_total}"
            if self.roundtrip_total
            else "skipped"
        )
        return (
            f"{self.index_type}(metric={self.metric}) ntotal={self.ntotal:,} d={self.dim} | "
            f"id={self.id_min}..{self.id_max} base={self.id_base} "
            f"permutation={self.id_is_permutation} row_aligned={self.id_row_aligned}"
            f"(misaligned={self.n_row_misaligned:,}) | "
            f"json_matches={self.json_matches_index} | roundtrip={rt} | ok={self.ok}"
        )


def load_id_map(path: Path) -> dict[str, int]:
    """arxiv_id -> FAISS id. 값은 1-based입니다."""
    with open(path) as f:
        return {k: int(v) for k, v in json.load(f).items()}


def _metric_name(index) -> str:
    inner = faiss.downcast_index(index.index) if hasattr(index, "index") else index
    return "IP" if inner.metric_type == faiss.METRIC_INNER_PRODUCT else "L2"


def inspect_mapping(index, id_to_index: dict[str, int]) -> MappingReport:
    """구조 수준 검증 — GPU도 임베딩 모델도 필요 없습니다.

    두 가지를 **구분해서** 봅니다. 섞어 보면 틀립니다:

    - `id_is_permutation`: id 집합이 {1..N} 인가 → 참. 검색·조회의 정합성은 이것만 있으면 됩니다
    - `id_row_aligned`: id_map[row] == row+1 인가 → **거짓**. 인덱스가 세 덩어리를
      다른 순서로 이어 붙여 만들어졌습니다. 즉 **`faiss_id - 1` 은 행 번호가 아닙니다.**
    """
    ids = faiss.vector_to_array(index.id_map)
    base = int(ids.min())
    expected = np.arange(base, base + len(ids), dtype=ids.dtype)

    row_aligned = bool(np.array_equal(ids, expected))
    n_misaligned = int(np.count_nonzero(ids != expected))
    is_perm = bool(np.array_equal(np.sort(ids), expected))

    vals = np.fromiter(id_to_index.values(), dtype=np.int64, count=len(id_to_index))
    matches = bool(len(vals) == len(ids) and np.array_equal(np.sort(vals), np.sort(ids)))

    return MappingReport(
        index_type=type(index).__name__,
        metric=_metric_name(index),
        ntotal=int(index.ntotal),
        dim=int(index.d),
        id_min=int(ids.min()),
        id_max=int(ids.max()),
        id_is_permutation=is_perm,
        id_row_aligned=row_aligned,
        n_row_misaligned=n_misaligned,
        id_base=base,
        json_entries=len(id_to_index),
        json_min=int(vals.min()),
        json_max=int(vals.max()),
        json_matches_index=matches,
    )


def build_id_to_row(index) -> dict[int, int]:
    """FAISS id -> 내부 행 번호.

    **이걸 거치지 않고 `faiss_id - 1` 을 행 번호로 쓰면 엉뚱한 벡터가 나옵니다.**
    id_map 이 순열이라 행 순서와 id 순서가 다르기 때문입니다(441,842개가 어긋나 있음).

    저장된 벡터를 꺼내야 하는 곳은 S7(MMR)입니다 — 초록 임베딩 간 유사도를 재려면
    쿼리 임베딩이 아니라 인덱스에 든 문서 벡터가 필요합니다. 그때 이 함수를 쓰세요:

        rows = build_id_to_row(index)
        inner = faiss.downcast_index(index.index)
        vec = inner.reconstruct(rows[faiss_id])       # O
        vec = inner.reconstruct(faiss_id - 1)          # X — 다른 논문의 벡터
    """
    ids = faiss.vector_to_array(index.id_map)
    return {int(fid): row for row, fid in enumerate(ids)}


def reconstruct_by_id(index, faiss_ids: list[int], id_to_row: dict[int, int] | None = None) -> np.ndarray:
    """FAISS id 목록에 해당하는 저장된 문서 벡터를 꺼냅니다 (행 번호 함정 회피)."""
    rows = id_to_row if id_to_row is not None else build_id_to_row(index)
    inner = faiss.downcast_index(index.index)
    return np.vstack([inner.reconstruct(rows[int(i)]) for i in faiss_ids])


def load_tinydb(path: Path) -> dict[str, dict]:
    """TinyDB JSON → {arxiv_id: record}. 1.4GB 파싱에 15~25초 걸립니다."""
    with open(path) as f:
        table = json.load(f)["cs_paper_info"]
    return {rec["id"]: rec for rec in table.values()}


def probe_roundtrip(
    index,
    id_to_index: dict[str, int],
    by_arxiv: dict[str, dict],
    n_probe: int = 10,
    device: str = "cuda",
) -> list[dict]:
    """의미 수준 왕복 — 논문 → 임베딩 → 검색 → 같은 논문이 1위인가.

    코퍼스 전체에 고르게 퍼지도록 균등 간격으로 뽑습니다 (특정 연도에 몰리면
    매핑이 구간별로 어긋나는 경우를 놓칩니다).
    """
    import torch
    from sentence_transformers import SentenceTransformer

    index_to_id = {v: k for k, v in id_to_index.items()}
    keys = list(by_arxiv)
    step = max(1, len(keys) // n_probe)
    probes = keys[::step][:n_probe]

    model = SentenceTransformer(GTE_MODEL, trust_remote_code=True)
    model.to(torch.device(device))
    texts = [f"{by_arxiv[p]['title']}\n{by_arxiv[p].get('abs', '')}" for p in probes]
    qv = np.asarray(model.encode(texts), dtype="float32")

    _, ids = index.search(qv, 3)
    out = []
    for arxiv_id, row in zip(probes, ids):
        got_id = int(row[0])
        got_arxiv = index_to_id.get(got_id)
        rec = {
            "arxiv_id": arxiv_id,
            "expected_faiss_id": id_to_index[arxiv_id],
            "got_faiss_id": got_id,
            "got_arxiv_id": got_arxiv,
            "ok": got_arxiv == arxiv_id,
        }
        if not rec["ok"]:
            # 가장 흔한 실패는 off-by-one 입니다. 진단을 같이 남깁니다.
            rec["neighbors"] = {
                "id-1": index_to_id.get(got_id - 1),
                "id+1": index_to_id.get(got_id + 1),
            }
        out.append(rec)
    return out


def measure_latency(index, n_queries: tuple[int, ...] = (1, 32), top_k: int = 1500) -> dict:
    """랜덤 단위벡터로 배치 크기별 검색 지연 측정 (P0.3).

    실제 쿼리가 아니라 랜덤 벡터인 이유: 지연은 브루트포스 스캔에 지배되므로
    쿼리 내용과 무관하고, 임베딩 모델 없이 잴 수 있어야 하기 때문입니다.
    """
    rng = np.random.default_rng(0)
    out = {}
    for n in n_queries:
        q = rng.normal(size=(n, index.d)).astype("float32")
        faiss.normalize_L2(q)
        index.search(q, top_k)  # 워밍업 (첫 호출에 페이지 폴트가 섞입니다)
        t = time.perf_counter()
        index.search(q, top_k)
        dt = time.perf_counter() - t
        out[n] = {"total_s": round(dt, 3), "per_query_ms": round(dt / n * 1000, 1)}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=Path, default=SURVEYFORGE.faiss_title_abs)
    ap.add_argument("--id-map", type=Path, default=SURVEYFORGE.id_map)
    ap.add_argument("--db", type=Path, default=SURVEYFORGE.tinydb)
    ap.add_argument("--probe", type=int, default=10, help="의미 수준 왕복 검증 논문 수")
    ap.add_argument("--skip-semantic", action="store_true", help="GPU/임베딩 없이 구조만 검증")
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json-out", type=Path, help="리포트를 JSON으로 저장")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    t = time.perf_counter()
    index = faiss.read_index(str(args.index))
    log.info("index loaded in %.1fs: %s", time.perf_counter() - t, args.index.name)

    id_to_index = load_id_map(args.id_map)
    report = inspect_mapping(index, id_to_index)

    if not args.skip_semantic:
        log.info("loading TinyDB (1.4GB, ~20s)...")
        t = time.perf_counter()
        by_arxiv = load_tinydb(args.db)
        log.info("TinyDB loaded in %.1fs: %d papers", time.perf_counter() - t, len(by_arxiv))
        detail = probe_roundtrip(index, id_to_index, by_arxiv, args.probe, args.device)
        report.roundtrip_detail = detail
        report.roundtrip_total = len(detail)
        report.roundtrip_ok = sum(d["ok"] for d in detail)
        for d in detail:
            log.info(
                "  %-14s expect=%-7d got=%-7d -> %-14s %s",
                d["arxiv_id"], d["expected_faiss_id"], d["got_faiss_id"],
                d["got_arxiv_id"], "OK" if d["ok"] else f"MISMATCH {d.get('neighbors')}",
            )

    latency = {}
    if not args.skip_latency:
        latency = measure_latency(index)
        for n, v in latency.items():
            log.info("latency: %2d queries top-1500 -> %.2fs (%.0f ms/query)",
                     n, v["total_s"], v["per_query_ms"])

    print(report.summary())

    if report.ntotal != N_PAPERS:
        log.warning("ntotal %d != 기록된 %d — SETTING.md 를 갱신하세요", report.ntotal, N_PAPERS)
    if report.dim != EMBED_DIM:
        log.warning("dim %d != 기록된 %d", report.dim, EMBED_DIM)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(
            {"mapping": report.__dict__, "latency": latency}, indent=2, default=str))
        log.info("report -> %s", args.json_out)

    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
