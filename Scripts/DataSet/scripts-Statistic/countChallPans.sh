grep -i "chall" ../lists/* | grep "$1" | cut -d':' -f1 | while read a ; do tail -n+6 $a | grep -iv none ; done | wc -l
