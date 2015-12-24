rm pairs.tmp
ls $1/*.ant | while read a ; do tail -n+4 $a  | cut -f12,13 >> ./pairs.tmp ; done
cat pairs.tmp | sed 's/0.*$//' | cut -f2 | grep -i 1 | wc -l
rm pairs.tmp
