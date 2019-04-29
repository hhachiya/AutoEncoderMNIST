#!/bin/bash

for zDim in 100 200; do
	for ((tri=0; tri<1; tri++)); do
		for ((char=0; char<10; char++)); do
			for noiseSigma in 0.0 0.2 0.5; do
				python adversarialClassifierAug.py 0 $char $tri $noiseSigma $zDim 10000

				#for augRatio in 1 2 5; do
				#	python adversarialClassifierAug.py 1 $char $tri $noiseSigma $zDim 10000 $augRatio
				#done
			done
		done
	done
done
