grep -iv none ../lists/* | egrep --color -io "light\(.*\)" ../lists/* | cut -d',' -f1 | egrep --color "\([LNH]{2,3}\)" | wc -l
