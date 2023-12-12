#!/usr/bin/env bash

cd ../..

CONFIG=${CONFIG:-v3}
GRID=${GRID:-v3}
REPEAT=${REPEAT:-3}
MAX_JOBS=${MAX_JOBS:-2}
SLEEP=${SLEEP:-1}
MAIN=${MAIN:-main}

echo "Started job"

# generate configs (after controlling computational budget)
# please remove --config_budget, if don't control computational budget
#python configs_gen.py --config configs/MIGNN/${CONFIG}.yaml \
#  --grid grids/MIGNN/${GRID}.txt \
#  --out_dir configs
#python configs_gen.py --config configs/ChemKG/${CONFIG}.yaml --config_budget configs/ChemKG/${CONFIG}.yaml --grid grids/ChemKG/${GRID}.txt --out_dir configs
# run batch of configs
# Args: config_dir, num of repeats, max jobs running, sleep time
bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP $MAIN
# rerun missed / stopped experiments
bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP $MAIN
# rerun missed / stopped experiments
bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP $MAIN

# aggregate results for the batch
python agg_batch.py --dir results/${CONFIG}_grid_${GRID}
