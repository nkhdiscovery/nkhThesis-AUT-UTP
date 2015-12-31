#!/bin/bash
rm -rf ./nkhTmpNoHead/*.nkhtmp
frames=`cat "$1" | cut -f11 | tail -1` 
frames=$((++frames))
objects=`cat "$1" | cut -f4 | uniq | wc -l`
for i in `seq $objects` 
do
    #echo $i
    skipTail=$((--i*frames+1))
    tail -n+$skipTail "$1" | head -$frames | awk '$12 == 1 { print $0 }' | cut -f2- | sort -k 10,10 > "$1.$i.nkhtmp"
done 

if [ $objects -le 1 ]; then
    cat "$1" | cut -f11 > ./"$1".1.tmp
    cat "$1" | cut -f2-10,12,13 > ./"$1".2.tmp
    paste ./"$1".1.tmp ./"$1".2.tmp > "$1".merged
else
    ./recJoin.sh nkhTmpNoHead/*.nkhtmp | sort -n -k 1,1 > "$1".merged
fi
