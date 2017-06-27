#!/bin/bash
#please change it to your own filepath
videopre=../../IsoGD_phase_1/
flowipost=/flow_i/flow_i
flowxpost=/flow_x/flow_x
flowypost=/flow_y/flow_y
count=0
cat valid_filename | while read line
do 
videofile=${videopre}${line}
flowfile1=${line:0:10}
flowfile2=${line:12:5}
flowfilepre=${flowfile1}${flowfile2}
flowfilei=${flowfilepre}${flowipost}
flowfilex=${flowfilepre}${flowxpost}
flowfiley=${flowfilepre}${flowypost}
./denseFlow_gpu -f ${videofile} -x ${flowfilex} -y ${flowfiley} -i ${flowfilei} -b 20 -t 1 -d 0 -s 1
count=$((${count}+1))
if [ $((${count} % 100)) == 0 ]
then 
  echo ${count}
fi
done

