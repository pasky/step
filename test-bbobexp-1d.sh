#!/bin/sh
# Run test.py on a set of functions in 1D, collecting the BBOB fev data
# (and final averaged results).
# Usage: ./test-bbobexp-1d.sh ALIAS TEST_PARAM...
# Example: ./test-bbobexp-1d.sh ndstep ndstep
# Example: ./test-bbobexp-1d.sh ndsqistept500 -t 500 ndsqistep

alias="$1"
shift
if [ -z "$alias" ]; then
	echo "Usage: ./test-bbobexp-1d.sh ALIAS TEST_PARAM..." >&2
	exit 1
fi

for i in `seq 1 24`; do
	mkdir -p 1d-data/B$i
	echo B$i
	./test.py -d 1 -f B$i -i 10000 "$@" | tee 1d-data/B$i/"$alias"
done

grep converg 1d-data/B*/"$alias"
