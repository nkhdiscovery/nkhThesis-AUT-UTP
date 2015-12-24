grep -iv none ../lists/* | grep -i "side_" | egrep --color -io "color\(.*\)"  | cut -d',' -f1 | egrep --color "\($1\)" | wc -l 
