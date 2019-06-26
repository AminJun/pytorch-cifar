#!/bin/bash
#python graph.py -f test_0.xls test_1.xls -l 'Boring' 'Fast' -o test.pdf -d outs/
#python graph.py -f train_0.xls train_1.xls -l 'Boring' 'Fast' -o train.pdf -d outs/
python graph.py -f Train_boring.xls Train_fast.xls -l 'Boring' 'Fast' -o train_100 -d .
python graph.py -f Test_boring.xls Test_fast.xls -l 'Boring' 'Fast' -o Test_100 -d .
