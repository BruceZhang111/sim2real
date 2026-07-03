#!/usr/bin/env bash
# (Re)download the SO-101 MuJoCo model + meshes from TheRobotStudio/SO-ARM100.
# The model is vendored under assets/so101/ so the repo is self-contained; run
# this only to re-sync with upstream.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST="${HERE}/assets/so101"
BASE="https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/main/Simulation/SO101"

mkdir -p "${DST}/assets"

echo ">> fetching MJCF"
for f in scene.xml so101_new_calib.xml joints_properties.xml; do
  curl -sSfL "${BASE}/${f}" -o "${DST}/${f}"
done

echo ">> fetching meshes"
MESHES="waveshare_mounting_plate_so101_v2 sts3215_03a_v1 motor_holder_so101_base_v1 \
wrist_roll_follower_so101_v1 moving_jaw_so101_v1 base_motor_holder_so101_v1 \
upper_arm_so101_v1 wrist_roll_pitch_so101_v2 under_arm_so101_v1 rotation_pitch_so101_v1 \
motor_holder_so101_wrist_v1 sts3215_03a_no_horn_v1 base_so101_v2"
for m in $MESHES; do
  curl -sSfL "${BASE}/assets/${m}.stl" -o "${DST}/assets/${m}.stl"
done

echo ">> done. Vendored $(ls "${DST}/assets" | wc -l) meshes into ${DST}/assets"
echo ">> reach_scene.xml (the task scene wrapping the robot) is kept in-repo and not overwritten."
