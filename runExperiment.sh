#!/bin/bash

for zDim in 10 100; do
	for ((tri=0; tri<1; tri++)); do
		for ((char=0; char<10; char++)); do
			for noiseSigma in 0 25; do
				python adversarialClassifierAug.py 0 $char $tri $noiseSigma $zDim 10000

				for noiseSigmaEmbed in 2 5; do
					python adversarialClassifierAug.py 6 $char $tri $noiseSigma $zDim 10000 $noiseSigmaEmbed
				done
			done
		done
	done
done
