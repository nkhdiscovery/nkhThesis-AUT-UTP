#!/bin/bash
rm -rf nkhTmpNoHead
mkdir nkhTmpNoHead
ls "$1" | grep -i "\.ant" | while read a ;do tail -n+4 "$1"/$a > ./nkhTmpNoHead/"$a".noHead ; ./mergeSingleAnt.sh ./nkhTmpNoHead/"$a".noHead ; mv ./nkhTmpNoHead/"$a".noHead.merged ./"$a".merged ; done
rm -rf nkhTmpNoHead
