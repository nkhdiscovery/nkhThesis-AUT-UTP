export AUTUTP_DIR_MP4="/media/navid/D410CF4E10CF3670/00Khazaee-DONotDelete-09132943946/DB/newcut-data/vids-1080"
ls $AUTUTP_DIR_MP4/*.mp4 | while read a ; do b=${a##*/}; name=`echo $b | sed 's/\.mp4//g'` ; echo -n "$a:-" >> $1; ../../build-processDir-Desktop-Release/processDir "$a" ../../DATA/merged/$name.ant.merged ../../test/out-alg/$name 0.5 300 300 >> $1; done
