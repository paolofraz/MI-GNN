CONFIG_DIR=$1
REPEAT=$2
MAX_JOBS=$3
SLEEP=$4
MAIN=$5

echo "parallel.sh: $1 - $2 - $3 - $4 - $5"
COUNTER=0

(
  trap 'kill 0' SIGINT
  CUR_JOBS=0
  for CONFIG in "$CONFIG_DIR"/*.yaml; do
    if [ "$CONFIG" != "$CONFIG_DIR/*.yaml" ]; then
      ((CUR_JOBS >= MAX_JOBS)) && wait -n
      ((++COUNTER))
      echo "#$COUNTER job launched: $CONFIG"
      python $MAIN.py --cfg $CONFIG --repeat $REPEAT --mark_done &
      #((CUR_JOBS < MAX_JOBS)) && sleep 1
      sleep $SLEEP
      ((++CUR_JOBS))
    fi
  done

  wait
)
