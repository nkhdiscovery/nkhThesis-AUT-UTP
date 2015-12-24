(find $1 -iname "*.mp4" | while read a ; do ffprobe -select_streams v -show_streams $a 2>/dev/null | grep nb_frames | sed -e 's/nb_frames=//' ; done) > $2
echo $(cat $2 | tr '\n' '+')0 | bc 
