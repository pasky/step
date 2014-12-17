#!/bin/sh
# Run test.py on a set of functions in 3D and 5D, collecting the final
# averaged results.
# Example: ./test.sh -r rdiffpd ndstep

{
for d in 3 5; do
	for b in 2 3 4 5 7; do
		./test.py -d $d -f b$b -r 10 "$@" | sed 's/.*converged/b'$b',d'$d'\t&/'
	done
done
} | tee /tmp/ndstep-test.$$
grep converged /tmp/ndstep-test.$$ | sed 's/^/'"$*"':\t/'
