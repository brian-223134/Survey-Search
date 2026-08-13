#!/usr/bin/env bash
# 170토픽 평가의 감시자. 죽거나 서면 다시 띄웁니다 — 체크포인트가 있어서 이어서 돕니다.
#
# 체크포인트가 없던 시절엔 재시작이 곧 처음부터라 감시자를 둘 수 없었습니다.
# 이제 (설정,토픽) 쌍 단위로 남으니 죽은 지점부터 잇고, 무한 재시작만 막으면 됩니다.
#
# 두 가지를 봅니다. **죽음**은 60초 안에(프로세스 유무), **정지**는 체크포인트가
# STALL_MIN 분간 한 줄도 안 자라는 것으로. 둘을 같은 주기로 묶으면 죽음을 늦게
# 알아채거나 정상적인 느린 토픽을 정지로 오인합니다.
#
#   setsid nohup scripts/watch_surge_eval.sh > data/logs/surge170_watchdog.log 2>&1 &
set -u

CKPT=data/surge_eval_170.ckpt.jsonl
OUT=data/surge_eval_170.json
LOG=data/logs/surge170.log
# **cmdline 전체에 앵커를 겁니다.** 그냥 "python scripts/eval_surge_full.py" 로 찾으면
# 그 문자열을 인자로 가진 셸 래퍼(`bash -c '... pgrep -f "python scripts/eval_surge_full.py"'`)
# 까지 잡힙니다. 예전에 이 오탐 때문에 죽은 프로세스를 살아 있다고 보고한 적이 있습니다.
PAT="^[^ ]*python scripts/eval_surge_full\.py$"
MAX_RESTARTS=${MAX_RESTARTS:-5}
STALL_MIN=${STALL_MIN:-25}
POLL=${POLL:-60}

say() { echo "$(date '+%F %T') watchdog: $*"; }
lines() { wc -l < "$CKPT" 2>/dev/null || echo 0; }

restarts=0
last=$(lines)
grew_at=$(date +%s)
say "감시 시작 (체크포인트 ${last}줄, 정지판정 ${STALL_MIN}분, 폴링 ${POLL}초)"

while :; do
  if [ -f "$OUT" ]; then say "결과 파일 있음 -> 완료. 감시 종료"; exit 0; fi

  now=$(date +%s)
  cur=$(lines)
  if [ "$cur" -gt "$last" ]; then last=$cur; grew_at=$now; fi

  if pgrep -f "$PAT" > /dev/null; then
    idle=$(( (now - grew_at) / 60 ))
    if [ "$idle" -ge "$STALL_MIN" ]; then
      say "정지 의심 — 살아 있는데 ${idle}분간 체크포인트 ${cur}줄 그대로. 죽이고 이어받습니다"
      pkill -f "$PAT"; sleep 10; grew_at=$(date +%s)
    fi
    sleep "$POLL"; continue
  fi

  # 안 돌고 있는데 결과도 없음 = 죽었습니다.
  if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
    say "재시작 $MAX_RESTARTS 회를 다 썼습니다. 사람이 봐야 합니다 (체크포인트 ${cur}줄)"
    exit 1
  fi
  restarts=$((restarts + 1))
  say "프로세스 없음 -> 재시작 $restarts/$MAX_RESTARTS (체크포인트 ${cur}줄에서 이어받음)"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6} setsid nohup \
    .venv/bin/python scripts/eval_surge_full.py >> "$LOG" 2>&1 < /dev/null &
  grew_at=$(date +%s)
  sleep "$POLL"
done
