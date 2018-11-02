#!/bin/bash

for zDim in 100; do
	for ((tri=0; tri<5; tri++)); do
		for ((char=0; char<10; char++)); do
			#for noiseSigmaEmbed in 2 3 5; do
			for noiseSigma in 10 50 100; do
				#python adversarialClassifierAug.py 5 $char $tri $noiseSigma $zDim 5000 $noiseSigmaEmbed
				#python adversarialClassifierAug.py 1 $char $tri $noiseSigma $zDim 5000 10
				python adversarialClassifierAug.py 0 $char $tri $noiseSigma $zDim 10000
			done
		done
	done
done
