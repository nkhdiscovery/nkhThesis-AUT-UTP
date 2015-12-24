(ls $1/*.ant | while read vid ; do rm *.nkh ; tail -n+4 $1/$vid | cut -f4 | uniq | while read a ; do grep -i $a$'\t' $1/$vid | cut -f12 > ./$a.nkh ; done ; paste *.nkh | grep -i 1 | wc -l ; done) >./pos.tmp
echo $(cat pos.tmp | tr '\n' '+')0 | bc 
