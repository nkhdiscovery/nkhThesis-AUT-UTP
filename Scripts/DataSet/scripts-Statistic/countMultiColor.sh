grep -iv none ../lists/* | egrep --color -io "color\(.*\)" ../lists/* | cut -d',' -f1 | egrep --color "\([WGBO]{3}\)" | wc -l
