#!/bin/bash 

for t in 'train' 'test' ; do 
	for w in '0' '1' ; do 
		file="${t}_${w}.xls"
		rm "${file}"
	done
done

for e in `seq 10 10 200` ; do 
	for w in '0' '1' ; do 
		f="${w}_${e}.out"
		t=$(tail -n 1 "${f}" | awk '{print $3}')		
		tr=$(tail -n 6 "${f}" | grep Train | tail -n 1 | awk '{print $3}')
		ts=$(tail -n 6 "${f}" | grep Test | tail -n 1 | awk '{print $3}')
		echo -e "${t}\t${tr}" >> train_${w}.xls
		echo -e "${t}\t${ts}" >> test_${w}.xls
	done
done

for t in 'train' 'test' ; do 
	for w in '0' '1' ; do 
		file="${t}_${w}.xls"
		echo "======${file}====="
		cat "${file}"
	done
done
