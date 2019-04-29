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

#-------------
z_dim_R = 100
trainMode = 0
augRatio = 1
noiseSigma = 0.5

# trial numbers
trialNos = [0]

# Iteration
nIte = 10000
resInd = int((nIte-1)/1000)
#-------------

#-------------
# Characters
targetChars = [0,1,2,3,4,5,6,7,8,9]

# テストデータにおける偽物の割合
testAbnormalRatios = [0.1, 0.2, 0.3, 0.4, 0.5]
#-------------

#-------------
# Path
logPath = 'logs'

# Methods
ALOCC = 0
TRIPLE = 1
#-------------

#-------------
# methods
if trainMode == ALOCC:
	postFixStr = 'ALOCC'
elif trainMode == TRIPLE:
	postFixStr = 'TRIPLE'
#-------------

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
		predictX_value = pickle.load(fp)
		predictRX_value = pickle.load(fp)

		recallXs = pickle.load(fp)
		precisionXs = pickle.load(fp)
		f1Xs = pickle.load(fp)
		aucXs = pickle.load(fp)
		recallRXs = pickle.load(fp)
		precisionRXs = pickle.load(fp)
		f1RXs = pickle.load(fp)
		aucRXs = pickle.load(fp)

		if trainMode >= TRIPLE:
			recallXs = pickle.load(fp)
			precisionXs = pickle.load(fp)
			f1Xs = pickle.load(fp)
			aucXs = pickle.load(fp)
			recallRXs = pickle.load(fp)
			precisionRXs = pickle.load(fp)
			f1RXs = pickle.load(fp)
			aucRXs = pickle.load(fp)	

		lossR_values = pickle.load(fp)
		lossRAll_values = pickle.load(fp)
		lossD_values = pickle.load(fp)

		if trainMode >= TRIPLE:
			lossC_values = pickle.load(fp)
			lossA_values = pickle.load(fp)
			decoderR_train_abnormal_value = pickle.load(fp)
		else:
			lossC_values = []
			lossA_values = []
			decoderR_train_abnormal_value = []

		params = pickle.load(fp)	

		return recallXs, precisionXs, f1Xs, aucXs, recallRXs, precisionRXs, f1RXs, aucRXs, lossR_values, lossRAll_values, lossD_values, encoderR_train_value, lossC_values, lossA_values
#===========================

#===========================
recallXs = [[] for tmp in targetChars]
precisionXs = [[] for tmp in targetChars]
f1Xs = [[] for tmp in targetChars]
aucXs = [[] for tmp in targetChars]
recallRXs = [[] for tmp in targetChars]
precisionRXs = [[] for tmp in targetChars]
f1RXs = [[] for tmp in targetChars]
aucRXs = [[] for tmp in targetChars]
lossR_values = [[] for tmp in targetChars]
lossRAll_values = [[] for tmp in targetChars]
lossD_values = [[] for tmp in targetChars]
lossC_values = [[] for tmp in targetChars]
lossA_values = [[] for tmp in targetChars]
maxInds = [[] for tmp in targetChars]
#===========================

#===========================
# load pickles
for targetChar in targetChars:
	for trialNo in trialNos:
		# ファイル名のpostFix
		if trainMode == ALOCC:
			postFix = "_{}_{}_{}_{}_{}".format(postFixStr, targetChar, trialNo, z_dim_R, noiseSigma)
		elif trainMode == TRIPLE:
			postFix = "_{}_{}_{}_{}_{}_{}".format(postFixStr, targetChar, trialNo, z_dim_R, noiseSigma, augRatio)


		#--------------
		# pickleから読み込み
		path = os.path.join(logPath,"log{}.pickle".format(postFix))

		recallXs_, precisionXs_, f1Xs_, aucXs_, recallRXs_, precisionRXs_, f1RXs_, aucRXs_, lossR_values_, lossRAll_values_, lossD_values_, encoderR_train_value_, lossC_values_, lossA_values_ = loadParams(path)
		#--------------

		#--------------
		# 記録
		recallXs[targetChar].append(recallXs_)
		precisionXs[targetChar].append(precisionXs_)
		f1Xs[targetChar].append(f1Xs_)
		aucXs[targetChar].append(aucXs_)
		recallRXs[targetChar].append(recallRXs_)	
		precisionRXs[targetChar].append(precisionRXs_)
		f1RXs[targetChar].append(f1RXs_)
		aucRXs[targetChar].append(aucRXs_)
		lossR_values[targetChar].append(lossR_values_)
		lossRAll_values[targetChar].append(lossRAll_values_)
		lossD_values[targetChar].append(lossD_values_)
		lossC_values[targetChar].append(lossC_values_)
		lossA_values[targetChar].append(lossA_values_)
		#--------------

	#--------------
	# 最大のlossDに対応するF1 score 
	maxInds[targetChar] = np.argmax(np.array(f1Xs[targetChar])[:,-1,resInd])
	#--------------
#===========================

#===========================
# average evaluation
recalls = [[] for tmp in np.arange(len(targetChars))]
precisions = [[] for tmp in np.arange(len(targetChars))]
f1s = [[] for tmp in np.arange(len(targetChars))]
aucs = [[] for tmp in np.arange(len(targetChars))]
recallsR = [[] for tmp in np.arange(len(targetChars))]
precisionsR = [[] for tmp in np.arange(len(targetChars))]
f1sR = [[] for tmp in np.arange(len(targetChars))]
aucsR = [[] for tmp in np.arange(len(targetChars))]

for targetChar in targetChars:
	recalls_ = np.array(recallXs[targetChar][maxInds[targetChar]])[:,resInd]
	precisions_ = np.array(precisionXs[targetChar][maxInds[targetChar]])[:,resInd]
	f1s_ = np.array(f1Xs[targetChar][maxInds[targetChar]])[:,resInd]
	aucs_ = np.array(aucXs[targetChar][maxInds[targetChar]])[:,resInd]
	recallsR_ = np.array(recallRXs[targetChar][maxInds[targetChar]])[:,resInd]
	precisionsR_ = np.array(precisionRXs[targetChar][maxInds[targetChar]])[:,resInd]
	f1sR_ = np.array(f1RXs[targetChar][maxInds[targetChar]])[:,resInd]
	aucsR_ = np.array(aucRXs[targetChar][maxInds[targetChar]])[:,resInd]

	recalls[targetChar] = recalls_
	precisions[targetChar] = precisions_
	f1s[targetChar] = f1s_
	aucs[targetChar] = aucs_
	recallsR[targetChar] = recallsR_
	precisionsR[targetChar] = precisionsR_
	f1sR[targetChar] = f1sR_
	aucsR[targetChar] = aucsR_

recall_mean = np.mean(np.array(recalls),axis=0)
precision_mean = np.mean(np.array(precisions),axis=0)
f1_mean = np.mean(np.array(f1s),axis=0)
auc_mean = np.mean(np.array(aucs),axis=0)
recall_meanR = np.mean(np.array(recallsR),axis=0)
precision_meanR = np.mean(np.array(precisionsR),axis=0)
f1_meanR = np.mean(np.array(f1sR),axis=0)
auc_meanR = np.mean(np.array(aucsR),axis=0)
#===========================

#===========================
# print 
print("recall:",recall_mean)
print("precision:",precision_mean)
print("f1_mean:",f1_mean)
print("auc_mean:",auc_mean)
print('--------------')
print("recall R:",recall_meanR)
print("precision R:",precision_meanR)
print("f1 R:",f1_meanR)
print("auc R:",auc_meanR)
#===========================

pdb.set_trace()
