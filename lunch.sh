#!/bin/bash
file="scripts/${1}_${2}.sh"
cp "template.sh" "$file"
sed -i "s/FAST/$1/g" "${file}"
sed -i "s/EPOC/$2/g" "${file}"
jlunch "${file}"
