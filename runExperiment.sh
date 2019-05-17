#!/bin/bash

for zDim in 100 50; do
	for ((tri=0; tri<1; tri++)); do
		for ((char=0; char<10; char++)); do
			for noiseSigma in 0.155 0.0; do
				for stopTrainThre in 0.05 0.01; do
					python adversarialClassifierAug.py 0 $char $tri $noiseSigma $zDim 10000 $stopTrainThre

				#	python adversarialClassifierAug.py 2 $char $tri $noiseSigma $zDim 10000 $stopTrainThre 1
				done

				#python adversarialClassifierAug.py 1 $char $tri $noiseSigma $zDim 10000
			done
		done
	done
done
