#!/bin/bash
while true ; do 
p=$( echo -e "way\tepoch\ttrain\ttest"
for f in "boring" "fast" ; do 
	file="${f}.out"
	train=$(tail "${file}" | grep "Train" | tail -n 1 | awk '{print $3}')
	ttest=$(tail "${file}" | grep "Test"  | tail -n 1 | awk '{print $3}')
	epoch=$(tail "${file}" | grep "Epoch" | tail -n 1 | awk '{print $2}')
	echo -e "${f}\t${epoch}\t${train}\t${ttest}"
done)
	if [ "${lp}" != "${p}" ] ; then 
	echo -e "way\tepoch\ttrain\ttest"
for f in "boring" "fast" ; do 
	file="${f}.out"
	train=$(tail "${file}" | grep "Train" | tail -n 1 | awk '{print $3}')
	ttest=$(tail "${file}" | grep "Test"  | tail -n 1 | awk '{print $3}')
	epoch=$(tail "${file}" | grep "Epoch" | tail -n 1 | awk '{print $2}')
	echo -e "${f}\t${epoch}\t${train}\t${ttest}"
done
	lp="$p"
	sleep 1;
	echo -e "\n"

	fi	
done
