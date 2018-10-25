#!/bin/bash

for ((tri=0; tri<3; tri++)); do
	for noiseSigma in 10 50 100; do
		for ((char=0; char<10; char++)); do
			python adversarialClassifierAug.py 2 $char $tri $noiseSigma
			#python adversarialClassifierAug.py 1 $char $tri $noiseSigma
			#python adversarialClassifierAug.py 0 $char $tri $noiseSigma
		done
	done
done

