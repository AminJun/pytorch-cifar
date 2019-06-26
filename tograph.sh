#!/bin/bash
for i in 'boring' 'fast' ; do 
for j in 'Train' 'Test' ; do 
for e in '_e_' '_' ; do 
	f="${j}${e}${i}.xls"
	rm ${f};
done 
done
done 
for w in 'boring' 'fast' ; do 
	f="${w}.out"
	len=$(cat "${f}" | grep "Test" | wc -l) 
	for i in `seq 1 ${len}` ; do 
		t=$(cat "${f}" | grep "Time" | tail -n "${i}" | head -n 1 | awk '{print $3}')
		e=$(cat "${f}" | grep "Train" | tail -n "${i}" | head -n 1 | awk '{print $2}')
		tr=$(cat "${f}" | grep "Train" | tail -n "${i}" | head -n 1 | awk '{print $3}')
		ts=$(cat "${f}" | grep "Test" | tail -n "${i}" | head -n 1 | awk '{print $3}')
		echo -e "$t\t$tr\t$ts"
		echo -e "${t}\t${tr}" >> "Train_${w}.xls"
		echo -e "${t}\t${ts}" >> "Test_${w}.xls"
		echo -e "${e}\t${tr}" >> "Train_e_${w}.xls"
		echo -e "${e}\t${ts}" >> "Test_e_${w}.xls"
	done
		
done

ls .

for i in 'boring' 'fast' ; do 
for j in 'Train' 'Test' ; do 
for e in '_e_' '_' ; do 
	f="${j}${e}${i}.xls"
	echo "======+${f}======="
	cat "${f}"
done 
done 
done 
source graph.sh
