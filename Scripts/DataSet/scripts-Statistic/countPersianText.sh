echo $(ls ../ants/*.ant | while read a ; do grep -i "persian_" ../ants/$a | cut -f4 | uniq | wc -l ; done  | tr '\n' '+')0 | bc 
