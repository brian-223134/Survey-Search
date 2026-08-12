"""S5 — 중복 제거. DESIGN §S5.

두 가지를 병합합니다:

1. **arXiv 버전 병합** — `2401.12345v1` 과 `v2` 는 같은 논문. 최신 버전을 대표로
   남기고 점수는 최대값을 씁니다 (버전이 갈려 순위가 반토막 나는 것을 막습니다)
2. **제목 정규화 일치** — 소문자화·영숫자만 남긴 뒤 같으면 병합 (재게시·크로스리스트)

실측: 이 코퍼스에 정규화 제목이 겹치는 그룹이 **806개** 있습니다. 죽은 코드가 아닙니다.

버린 건수는 반드시 세어서 돌려줍니다 — 무음 폐기 금지.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_VERSION_SUFFIX = re.compile(r"v\d+$")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def strip_version(paper_id: str) -> str:
    return _VERSION_SUFFIX.sub("", paper_id)


def version_of(paper_id: str) -> int:
    """`2401.12345v3` -> 3. 버전 표기가 없으면 0."""
    m = _VERSION_SUFFIX.search(paper_id)
    return int(m.group()[1:]) if m else 0


def normalize_title(title: str) -> str:
    """소문자화 + 영숫자 외 제거. 줄바꿈이 섞인 제목이 실제로 많습니다."""
    return _NON_ALNUM.sub("", (title or "").lower())


def dedup(
    scored: Sequence[tuple[str, float]],
    titles: dict[str, str] | None = None,
) -> tuple[list[tuple[str, float]], dict[str, int], dict[str, list[str]]]:
    """`(paper_id, score)` 목록에서 중복 제거.

    Returns:
        (남은 목록, 사유별 폐기 건수, 대표 id -> 흡수된 id 목록)

    `titles` 를 주지 않으면 제목 병합은 건너뜁니다. 건너뛴 사실이 폐기 건수 0 과
    구분되도록, 호출부에서 `titles is None` 여부를 stats 에 남기세요.
    """
    dropped = {"version": 0, "title": 0}
    merged_into: dict[str, list[str]] = {}

    # 1) 버전 병합 — base_id 별로 (최고점, 최신 버전) 대표를 고릅니다
    best: dict[str, tuple[str, float]] = {}
    for pid, score in scored:
        base = strip_version(pid)
        cur = best.get(base)
        if cur is None:
            best[base] = (pid, score)
            continue
        cur_pid, cur_score = cur
        # 대표는 최신 버전, 점수는 최대값 — 둘을 따로 정합니다
        rep = pid if version_of(pid) > version_of(cur_pid) else cur_pid
        loser = cur_pid if rep == pid else pid
        best[base] = (rep, max(score, cur_score))
        merged_into.setdefault(rep, []).append(loser)
        # 대표가 바뀌었으면 이전 대표가 흡수하던 것도 넘겨줍니다
        if rep != cur_pid and cur_pid in merged_into:
            merged_into[rep].extend(merged_into.pop(cur_pid))
        dropped["version"] += 1

    out = sorted(best.values(), key=lambda kv: (-kv[1], kv[0]))

    # 2) 제목 정규화 병합
    if titles is not None:
        seen_title: dict[str, str] = {}
        kept: list[tuple[str, float]] = []
        for pid, score in out:
            key = normalize_title(titles.get(pid, ""))
            if not key:              # 제목을 모르면 병합 대상에서 제외 (조용히 버리지 않음)
                kept.append((pid, score))
                continue
            rep = seen_title.get(key)
            if rep is None:
                seen_title[key] = pid
                kept.append((pid, score))
            else:
                merged_into.setdefault(rep, []).append(pid)
                dropped["title"] += 1
        out = kept

    return out, dropped, merged_into
