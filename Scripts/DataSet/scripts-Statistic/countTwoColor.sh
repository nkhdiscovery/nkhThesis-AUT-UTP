grep -iv none ../lists/* | egrep --color -io "color\(.*\)" ../lists/* | cut -d',' -f1 | egrep --color "\([WGBO]{2}\)" | wc -l
