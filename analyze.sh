#!/bin/bash
while true ; do 
	echo -e "way\tepoch\ttrain\ttest"
for f in "boring" "fast" ; do 
	file="${f}.out"
	train=$(tail "${file}" | head -n 5 | grep "Train" | tail -n 1)
	ttest=$(tail "${file}" | head -n 5 | grep "Test" | tail -n 1)
	epoch=$(tail "${file}" | grep "Epoch" | awk '{print $2}' | tail -n 1)
	echo -e "${f}\t${epoch}\t${train}\t${ttest}"
done
	sleep 1;
	echo -e "\n"
done
