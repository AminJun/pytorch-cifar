#!/bin/bash
while true ; do 
	echo -e "way\tepoch\ttrain\ttest"
for f in "boring" "fast" ; do 
	file="${f}.out"
	train=$(tail "${file}" | head -n 4 | grep "Train" | sed 's/.*Acc://g' | sed 's/\%.*//g')
	ttest=$(tail "${file}" | head -n 4 | grep "Test" | sed 's/.*Acc://g' | sed 's/\%.*//g')
	epoch=$(tail "${file}" | grep "Epoch" | awk '{print $2}' | tail -n 1)
	echo -e "${f}\t${epoch}\t${train}\t${ttest}"
done
	sleep 1;
	echo -e "\n"
done
