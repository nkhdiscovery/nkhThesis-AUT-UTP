grep -iv none ../lists/* | egrep --color -io "light\(.*\)" ../lists/* | cut -d',' -f1 | egrep --color "\($1\)" | wc -l 
