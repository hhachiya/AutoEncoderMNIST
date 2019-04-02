# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.ops import nn_ops
import numpy as np
import math, os
import pickle
import pdb
import input_data
import matplotlib.pylab as plt
import sys

#===========================
# パラメータの設定
z_dim_R = 100

targetChars = [0,1,2,3,4,5,6,7,8,9]

# Rの二乗誤差の重み係数
lambdaR = 0.4

# log(0)と0割防止用
lambdaSmall = 0.00001

# 予測結果に対する閾値
threFake = 0.5

# テストデータにおける偽物の割合
testFakeRatios = [0.1, 0.2, 0.3, 0.4, 0.5]

# trial numbers
trialNos = [0,1,2]

nIte = 5000
resInd = int((nIte-1)/1000)

# クラスタ数
clusterNum = 10


# Rの二乗誤差の閾値
threSquaredLoss = 200

# バッチデータ数
batch_size = 300

visualPath = 'visualization'
modelPath = 'models'
logPath = 'logs'

noiseSigmaEmbed = 2 
noiseSigma = 0

ALOCC = 0
ALDAD = 1
ALDAD2 = 2
ALDAD3 = 3
ALDAD4 = 4
ALDAD5 = 5

trainMode = 5

if trainMode == ALOCC:
	postFixStr = 'ALOCC'
elif trainMode == ALDAD:
	postFixStr = 'ALDAD'
elif trainMode == ALDAD2:
	postFixStr = 'ALDAD2'
elif trainMode == ALDAD3:
	postFixStr = 'ALDAD3'
elif trainMode == ALDAD4:
	postFixStr = 'ALDAD4'
elif trainMode == ALDAD5:
	postFixStr = 'ALDAD5'

#===========================

#===========================
# load data

def loadParams(path):
	with open(path, "rb") as fp:
		batch_x = pickle.load(fp)
		batch_x_fake = pickle.load(fp)
		encoderR_train_value = pickle.load(fp)
		decoderR_train_value = pickle.load(fp)
		predictFake_train_value = pickle.load(fp)
		predictTrue_train_value = pickle.load(fp)
		test_x = pickle.load(fp)
		test_y = pickle.load(fp)
		decoderR_test_value = pickle.load(fp)
		predictDX_value = pickle.load(fp)
		predictDRX_value = pickle.load(fp)
		recallDXs = pickle.load(fp)
		precisionDXs = pickle.load(fp)
		f1DXs = pickle.load(fp)
		recallDRXs = pickle.load(fp)
		precisionDRXs = pickle.load(fp)
		f1DRXs = pickle.load(fp)	
		lossR_values = pickle.load(fp)
		lossRAll_values = pickle.load(fp)
		lossD_values = pickle.load(fp)
		params = pickle.load(fp)	

		return recallDXs, precisionDXs, f1DXs, recallDRXs, precisionDRXs, f1DRXs, lossR_values, lossRAll_values, lossD_values, encoderR_train_value

recallDXs = [[] for tmp in targetChars]
precisionDXs = [[] for tmp in targetChars]
f1DXs = [[] for tmp in targetChars]
recallDRXs = [[] for tmp in targetChars]
precisionDRXs = [[] for tmp in targetChars]
f1DRXs = [[] for tmp in targetChars]
lossR_values = [[] for tmp in targetChars]
lossRAll_values = [[] for tmp in targetChars]
lossD_values = [[] for tmp in targetChars]
maxInds = [[] for tmp in targetChars]

for targetChar in targetChars:
	for trialNo in trialNos:
		# ファイル名のpostFix

		if trainMode == ALOCC:
			postFix = "_{}_{}_{}_{}_{}".format(postFixStr, targetChar, trialNo, z_dim_R, noiseSigma)
		elif trainMode > ALOCC:
			#postFix = "_{}_{}_{}_{}_{}_{}".format(postFixStr, targetChar, trialNo, z_dim_R, noiseSigma, noiseSigmaEmbed)
			postFix = "_{}_{}_{}_{}_{}_{}_{}".format(postFixStr, targetChar, trialNo, z_dim_R, noiseSigma, noiseSigmaEmbed, clusterNum)

		#--------------
		# pickleから読み込み
		path = os.path.join(logPath,"log{}.pickle".format(postFix))

		recallDXs_, precisionDXs_, f1DXs_, recallDRXs_, precisionDRXs_, f1DRXs_, lossR_values_, lossRAll_values_, lossD_values_, encoderR_train_value_ = loadParams(path)
		#--------------


		#--------------
		# 記録
		recallDXs[targetChar].append(recallDXs_)	
		precisionDXs[targetChar].append(precisionDXs_)
		f1DXs[targetChar].append(f1DXs_)
		recallDRXs[targetChar].append(recallDRXs_)	
		precisionDRXs[targetChar].append(precisionDRXs_)
		f1DRXs[targetChar].append(f1DRXs_)
		lossR_values[targetChar].append(lossR_values_)
		lossRAll_values[targetChar].append(lossRAll_values_)
		lossD_values[targetChar].append(lossD_values_)
		#--------------

	#--------------
	# 最大のlossDに対応するF1 score 
	#maxInds[targetChar] = np.argmax(np.array(lossD_values[targetChar])[:,nIte-1])
	maxInds[targetChar] = np.argmax(np.array(f1DXs[targetChar])[:,-1,resInd])
	#--------------
#===========================

recalls = [[] for tmp in np.arange(len(targetChars))]
precisions = [[] for tmp in np.arange(len(targetChars))]
f1s = [[] for tmp in np.arange(len(targetChars))]
recallsR = [[] for tmp in np.arange(len(targetChars))]
precisionsR = [[] for tmp in np.arange(len(targetChars))]
f1sR = [[] for tmp in np.arange(len(targetChars))]

for targetChar in targetChars:

	recalls_ = np.array(recallDXs[targetChar][maxInds[targetChar]])[:,resInd]
	precisions_ = np.array(precisionDXs[targetChar][maxInds[targetChar]])[:,resInd]
	f1s_ = np.array(f1DXs[targetChar][maxInds[targetChar]])[:,resInd]
	recallsR_ = np.array(recallDRXs[targetChar][maxInds[targetChar]])[:,resInd]
	precisionsR_ = np.array(precisionDRXs[targetChar][maxInds[targetChar]])[:,resInd]
	f1sR_ = np.array(f1DRXs[targetChar][maxInds[targetChar]])[:,resInd]

	recalls[targetChar] = recalls_
	precisions[targetChar] = precisions_
	f1s[targetChar] = f1s_
	recallsR[targetChar] = recallsR_
	precisionsR[targetChar] = precisionsR_
	f1sR[targetChar] = f1sR_

recall_mean = np.mean(np.array(recalls),axis=0)
precision_mean = np.mean(np.array(precisions),axis=0)
f1_mean = np.mean(np.array(f1s),axis=0)
recall_meanR = np.mean(np.array(recallsR),axis=0)
precision_meanR = np.mean(np.array(precisionsR),axis=0)
f1_meanR = np.mean(np.array(f1sR),axis=0)

print(recall_mean)
print(precision_mean)
print(f1_mean)
print('--------------')
print(recall_meanR)
print(precision_meanR)
print(f1_meanR)

