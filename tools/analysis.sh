# CONFIG=$1
# CHECKPOINT=$2
SAVEPATH=$1
VALANN=$2

CONFIG="${SAVEPATH}/${SAVEPATH##*/}.py"
CHECKPOINT="${SAVEPATH}/latest.pth"

echo $CONFIG
echo $CHECKPOINT
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --out "${SAVEPATH}/results.pkl" \
    --eval bbox
python $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --format-only \
    --options "jsonfile_prefix=${SAVEPATH}/results"
python $(dirname "$0")/analysis_tools/coco_error_analysis.py \
    "${SAVEPATH}/results.bbox.json" \
    "${SAVEPATH}/analysis" \
    --ann=$VALANN
python tools/analysis_tools/analyze_results.py \
    $CONFIG \
    "${SAVEPATH}/results.pkl" \
    "${SAVEPATH}/results" \
    --show-score-thr 0.3
python tools/analysis_tools/confusion_matrix.py \
    $CONFIG \
    "${SAVEPATH}/results.pkl" \
    "${SAVEPATH}/analysis" \
    --show