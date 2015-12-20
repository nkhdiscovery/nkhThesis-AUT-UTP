#!/bin/sh

# multijoin - join multiple files based on 10'th column

join_rec() {
    if [ $# -eq 1 ]; then
        join -a1 -a2 -1 1 -2 10 - "$1" #first file is already joint, so the matching col is the first col
    else
        f=$1; shift
        join -a1 -a2 -1 1 -2 10 - "$f" | join_rec "$@"
    fi
}

if [ $# -le 2 ]; then
    join -a1 -a2 -j10 "$@"
else
    f1=$1; f2=$2; shift 2
    join -a1 -a2 -j10 "$f1" "$f2" | join_rec "$@"
fi
