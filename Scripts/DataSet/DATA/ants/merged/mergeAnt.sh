#!/bin/bash
frames=`cat "$1" | cut -f11 | tail -1` 
lines=`wc -l "$1" | cut -f1`
echo $frames $lines
