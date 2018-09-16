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

if len(sys.argv) > 2:
	# 文字の種類
	targetChar = int(sys.argv[1])

	# テスト時のノイズの割合
	testFakeRatio = float(sys.argv[2])

else:
	# 文字の種類
	targetChar = 0

	# テスト時のノイズの割合
	testFakeRatio = 0.5

# Rの二乗誤差の重み係数
lambdaR = 0.4

# log(0)と0割防止用
lambdaSmall = 0.00001

# 予測結果に対する閾値
threFake = 0.5

# Rの二乗誤差の閾値
threSquaredLoss = 200

# ファイル名のpostFix
postFix = "_{}_{}".format(targetChar, testFakeRatio)

# バッチデータ数
batch_size = 300

# 変数をまとめたディクショナリ
params = {'z_dim_R':z_dim_R, 'testFakeRatio':testFakeRatio, 'labmdaR':lambdaR,
'threFake':threFake, 'targetChar':targetChar,'batch_size':batch_size}

trainMode = 1

visualPath = 'visualization'
modelPath = 'models'
logPath = 'logs'
#===========================

#--------------
# pickleに保存
path = os.path.join(logPath,"log{}.pickle".format(postFix))
with open(path, "rb") as fp:
	batch_x = pickle.load(fp)
	batch_x_fake = pickle.load(fp)
	encoderR_train_value = pickle.load(fp)
	decoderR_train_value = pickle.load(fp)
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
#--------------

pdb.set_trace()