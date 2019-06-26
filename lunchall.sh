for i in `seq 110 10 200` ; do 
	for t in '0' '1' ; do 
	source lunch.sh $t $i
done	
done
