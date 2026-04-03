#!/usr/bin/env sh
set -eu

ENV_NAME="DEIO"
REPO="/home/s/repos/DEIO"
CONFIG="$REPO/config/train_tracker_base.conf"

MODES="
diversity
multi_motion_consistency
event_teacher
info_gain_head
conditioning_head
forward_backward_cycle
replay_stability
cycle_stability
"

cd "$REPO"

for mode in $MODES; do
  conda run -n "$ENV_NAME" python "$REPO/train_tracker.py" \
    -c "$CONFIG" \
    --name "tracker_${mode}" \
    --score_objective "$mode" \
    --eval
done
