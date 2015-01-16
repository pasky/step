#!/bin/sh
# Run test.py on a set of functions in various dimensions, collecting
# the BBOB fev data (and final averaged results).
# Example: ./test-bbobexp.sh -e rdiffpd ndstep

{
for d in 2 3 5 10 20 40; do
	for b in 1 2 3 4 5 7; do
		if [ "$b" -eq 7 -a "$d" -gt 2 ]; then
			continue
		fi
		./test.py -d $d -f B$b -i $(($d*100000)) "$@" | sed 's/.*converged/b'$b',d'$d'\t&/'
	done
done
} | tee /tmp/ndstep-test.$$
grep converged /tmp/ndstep-test.$$ | sed 's/^/'"$*"':\t/'
