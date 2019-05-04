#!/bin/bash

for zDim in 100 50; do
	for ((tri=0; tri<1; tri++)); do
		for ((char=0; char<10; char++)); do
			for noiseSigma in 0.155 0.0; do
				#for stopTrainThre in 0.02 0.01; do
				for stopTrainThre in 0.05 0.06; do
					python adversarialClassifierAug.py 0 $char $tri $noiseSigma $zDim 10000 $stopTrainThre
				done

				for augRatio in 2; do
					python adversarialClassifierAug.py 1 $char $tri $noiseSigma $zDim 10000 $augRatio
				done
			done
		done
	done
done
