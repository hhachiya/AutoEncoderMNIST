# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.ops import nn_ops
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.distributions as tfd
from sklearn.metrics import roc_auc_score
import numpy as np
import math, os
import pickle
import pdb
#import input_data
import matplotlib.pylab as plt
import sys

# バッチデータ数
batchSize = 300
keepProbTrain = 0.8

#######################
# パラメータの設定

# 学習モード
ALOCC = 0
GAN = 1
TRIPLE = 2

isStop = False
isEmbedSampling = True
isTrain = True
isVisualize = True

# 文字の種類
trainMode = int(sys.argv[1])

augRatio = 1
stopTrainThre = 0.01

# trail no.
if len(sys.argv) > 2:
	targetChar = int(sys.argv[2])
	trialNo = int(sys.argv[3])
	noiseSigma = float(sys.argv[4])
	z_dim_R = int(sys.argv[5])
	nIte = int(sys.argv[6])
else:
	targetChar = 0
	trialNo = 0	
	noiseSigma = 40
	z_dim_R = 2
	nIte = 5000

if len(sys.argv) > 7:
	if trainMode == TRIPLE: # augment data
		augRatio = int(sys.argv[7])

	elif trainMode == ALOCC: # stopping Qriteria
		stopTrainThre = float(sys.argv[7])


# Rの二乗誤差の重み係数
alpha = 0.2

# log(0)と0割防止用
lambdaSmall = 10e-10

# テストデータにおける偽物の割合
testAbnormalRatios = [0.1, 0.2, 0.3, 0.4, 0.5]

# 予測結果に対する閾値
threAbnormal = 0.5

# Rの誤差の閾値
threLossR = 50

# Dの誤差の閾値
threLossD = -10e-8

# 変数をまとめたディクショナリ
params = {'z_dim_R':z_dim_R, 'testAbnormalRatios':testAbnormalRatios, 'labmdaR':alpha,
'threAbnormal':threAbnormal, 'targetChar':targetChar,'batchSize':batchSize}

# プロットする画像数
nPlotImg = 10

# ファイル名のpostFix
if trainMode == ALOCC:
	trainModeStr = 'ALOCC'	
	postFix = "_{}_{}_{}_{}_{}_{}".format(trainModeStr,targetChar, trialNo, z_dim_R, noiseSigma, stopTrainThre)

elif trainMode == GAN:
	trainModeStr = 'GAN'
	pdb.set_trace()
	postFix = "_{}_{}_{}_{}_{}".format(trainModeStr,targetChar, trialNo, z_dim_R, noiseSigma)
	
elif trainMode == TRIPLE:
	trainModeStr = 'TRIPLE'	
	postFix = "_{}_{}_{}_{}_{}_{}".format(trainModeStr,targetChar, trialNo, z_dim_R, noiseSigma, augRatio)

visualPath = 'visualization/'
modelPath = 'models/'
logPath = 'logs/'
#######################

#######################
# 評価値の計算用の関数
def calcEval(predict, gt, threAbnormal=0.5):

	auc = roc_auc_score(gt, predict)

	tmp_predict = np.zeros_like(predict)
	tmp_predict[predict >= threAbnormal] = 1.
	tmp_predict[predict < threAbnormal] = 0.

	recall = np.sum(predict[gt==1])/np.sum(gt==1)
	precision = np.sum(predict[gt==1])/np.sum(predict==1)
	f1 = 2 * (precision * recall)/(precision + recall)

	return recall, precision, f1, auc
#######################

#######################
# plot image
def plotImg(x,y,path):
	# 画像を保存
	plt.close()

	fig, figInds = plt.subplots(nrows=2, ncols=x.shape[0], sharex=True)

	for figInd in np.arange(x.shape[0]):
		fig0 = figInds[0][figInd].imshow(x[figInd,:,:,0],cmap="gray")
		fig1 = figInds[1][figInd].imshow(y[figInd,:,:,0],cmap="gray")

		# ticks, axisを隠す
		fig0.axes.get_xaxis().set_visible(False)
		fig0.axes.get_yaxis().set_visible(False)
		fig0.axes.get_xaxis().set_ticks([])
		fig0.axes.get_yaxis().set_ticks([])
		fig1.axes.get_xaxis().set_visible(False)
		fig1.axes.get_yaxis().set_visible(False)
		fig1.axes.get_xaxis().set_ticks([])
		fig1.axes.get_yaxis().set_ticks([])

	plt.savefig(path)
#######################

#######################
# レイヤーの関数
def weight_variable(name, shape):
	return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(stddev=0.1))
	
def bias_variable(name, shape):
	return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.1))

# batch normalization
def batch_norm(inputs,training, trainable=False):
	res = tf.layers.batch_normalization(inputs, training=training, trainable=training)
	return res
	
# 1D convolution layer
def conv1d_relu(inputs, w, b, stride):
	# tf.nn.conv1d(input,filter,strides,padding)
	#filter: [kernel, output_depth, input_depth]
	# padding='SAME' はゼロパティングしている
	conv = tf.nn.conv1d(inputs, w, stride, padding='SAME') + b
	conv = tf.nn.relu(conv)
	return conv

# 1D deconvolution
def conv1d_t_relu(inputs, w, b, output_shape, stride):
	conv = nn_ops.conv1d_transpose(inputs, w, output_shape=output_shape, stride=stride, padding='SAME') + b
	conv = tf.nn.relu(conv)
	return conv

# 2D convolution
def conv2d_relu(inputs, w, b, stride):
	# tf.nn.conv2d(input,filter,strides,padding)
	# filter: [kernel, output_depth, input_depth]
	# input 4次元([batch, in_height, in_width, in_channels])のテンソルを渡す
	# filter 畳込みでinputテンソルとの積和に使用するweightにあたる
	# stride （=１画素ずつではなく、数画素ずつフィルタの適用範囲を計算するための値)を指定
	# ただし指定は[1, stride, stride, 1]と先頭と最後は１固定とする
	conv = tf.nn.conv2d(inputs, w, strides=stride, padding='SAME') + b 
	conv = tf.nn.relu(conv)
	return conv

# 2D deconvolution layer
def conv2d_t_sigmoid(inputs, w, b, output_shape, stride):
	conv = tf.nn.conv2d_transpose(inputs, w, output_shape=output_shape, strides=stride, padding='SAME') + b
	conv = tf.nn.sigmoid(conv)
	return conv

# 2D deconvolution layer
def conv2d_t(inputs, w, b, output_shape, stride):
	conv = tf.nn.conv2d_transpose(inputs, w, output_shape=output_shape, strides=stride, padding='SAME') + b
	return conv

# 2D deconvolution layer
def conv2d_t_relu(inputs, w, b, output_shape, stride):
	conv = tf.nn.conv2d_transpose(inputs, w, output_shape=output_shape, strides=stride, padding='SAME') + b
	conv = tf.nn.relu(conv)
	return conv

# fc layer with ReLU
def fc_relu(inputs, w, b, keepProb=1.0):
	fc = tf.nn.dropout(inputs, keepProb)
	fc = tf.matmul(fc, w) + b
	fc = tf.nn.relu(fc)
	return fc

# fc layer
def fc(inputs, w, b, keepProb=1.0):
	fc = tf.nn.dropout(inputs, keepProb)
	fc = tf.matmul(fc, w) + b
	return fc
	
# fc layer with softmax
def fc_sigmoid(inputs, w, b, keepProb=1.0):
	fc = tf.nn.dropout(inputs, keepProb)
	fc = tf.matmul(fc, w) + b
	fc = tf.nn.sigmoid(fc)
	return fc

#######################

#######################
# エンコーダ
# 画像をz_dim次元のベクトルにエンコード
# reuse=Trueで再利用できる（tf.variable_scope() は，変数の管理に用いるスコープ定義）
def encoderR(x, z_dim, reuse=False, keepProb = 1.0, training=False):
	with tf.variable_scope('encoderR') as scope:
		if reuse:
			scope.reuse_variables()
	
		# padding='SAME'のとき、出力のサイズO = 入力サイズI/ストライドS
		# 28/2 = 14
		convW1 = weight_variable("convW1", [3, 3, 1, 4])
		convB1 = bias_variable("convB1", [4])
		conv1 = conv2d_relu(x, convW1, convB1, stride=[1,2,2,1])
		conv1 = batch_norm(conv1, training)
		
		# 14/2 = 7
		convW2 = weight_variable("convW2", [3, 3, 4, 16])
		convB2 = bias_variable("convB2", [16])
		conv2 = conv2d_relu(conv1, convW2, convB2, stride=[1,2,2,1])
		conv2 = batch_norm(conv2, training)

		#=======================
		# 特徴マップをembeddingベクトルに変換
		# 2次元画像を１次元に変更して全結合層へ渡す
		# np.prod で配列要素の積を算出
		conv2size = np.prod(conv2.get_shape().as_list()[1:])
		conv2 = tf.reshape(conv2, [-1, conv2size])
		
		# 7 x 7 x 32 -> z-dim*2
		fcW1 = weight_variable("fcW1", [conv2size, z_dim*2])
		fcB1 = bias_variable("fcB1", [z_dim*2])
		fc1 = fc_relu(conv2, fcW1, fcB1, keepProb)

	
		# z-dim*2 -> z-dim
		fcW2 = weight_variable("fcW2", [z_dim*2, z_dim])
		fcB2 = bias_variable("fcB2", [z_dim])
		#fc2 = fc_relu(fc1, fcW2, fcB2, keepProb)
		fc2 = fc(fc1, fcW2, fcB2, keepProb)
		#=======================

		return fc2
#######################

#######################
# デコーダ
# z_dim次元の画像にデコード
# reuse=Trueで再利用できる（tf.variable_scope() は，変数の管理に用いるスコープ定義）
def decoderR(z,z_dim,reuse=False, keepProb = 1.0, training=False):
	with tf.variable_scope('decoderR') as scope:
		if reuse:
			scope.reuse_variables()

		#=======================
		# embeddingベクトルを特徴マップに変換
		# 2次元画像を１次元に変更して全結合層へ渡す
		fcW1 = weight_variable("fcW1", [z_dim, z_dim*2])
		fcB1 = bias_variable("fcB1", [z_dim*2])
		fc1 = fc_relu(z, fcW1, fcB1, keepProb)

		fcW2 = weight_variable("fcW2", [z_dim*2, 7*7*16])
		fcB2 = bias_variable("fcB2", [7*7*16])
		fc2 = fc_relu(fc1, fcW2, fcB2, keepProb)

		batchSize = tf.shape(fc2)[0]
		fc2 = tf.reshape(fc2, tf.stack([batchSize, 7, 7, 16]))
		#=======================
		
		# padding='SAME'のとき、出力のサイズO = 入力サイズI/ストライドS
		# 7 x 2 = 14
		convW1 = weight_variable("convW1", [3, 3, 4, 16])
		convB1 = bias_variable("convB1", [4])
		conv1 = conv2d_t_relu(fc2, convW1, convB1, output_shape=[batchSize,14,14,4], stride=[1,2,2,1])
		conv1 = batch_norm(conv1, training)
		
		# 14 x 2 = 28
		convW2 = weight_variable("convW2", [3, 3, 1, 4])
		convB2 = bias_variable("convB2", [1])
		output = conv2d_t(conv1, convW2, convB2, output_shape=[batchSize,28,28,1], stride=[1,2,2,1])

		output = tf.nn.sigmoid(output)

		return output
#######################

#######################
# D Network
# 
# reuse=Trueで再利用できる（tf.variable_scope() は，変数の管理に用いるスコープ定義）
def DNet(x, out_dim=1, reuse=False, keepProb=1.0, training=False):
	with tf.variable_scope('DNet') as scope:
		if reuse:
			scope.reuse_variables()
	
		# padding='SAME'のとき、出力のサイズO = 入力サイズI/ストライドS
		# 28/2 = 14
		convW1 = weight_variable("convW1", [3, 3, 1, 4])
		convB1 = bias_variable("convB1", [4])

		conv1 = conv2d_relu(x, convW1, convB1, stride=[1,2,2,1])
		conv1 = batch_norm(conv1, training)
		
		# 14/2 = 7
		convW2 = weight_variable("convW2", [3, 3, 4, 16])
		convB2 = bias_variable("convB2", [16])
		
		conv2 = conv2d_relu(conv1, convW2, convB2, stride=[1,2,2,1])
		conv2 = batch_norm(conv2, training) 

		#=======================
		# 特徴マップをembeddingベクトルに変換
		# 2次元画像を１次元に変更して全結合層へ渡す
		# np.prod で配列要素の積を算出
		conv2size = np.prod(conv2.get_shape().as_list()[1:])
		conv2 = tf.reshape(conv2, [-1, conv2size])
		
		# 7 x 7 x 16 -> 128
		hidden_dim = 128
		fcW1 = weight_variable("fcW1", [conv2size, hidden_dim])
		fcB1 = bias_variable("fcB1", [hidden_dim])
		fc1 = fc_relu(conv2, fcW1, fcB1, keepProb)

		# 100 -> out_dim
		fcW2 = weight_variable("fcW2", [hidden_dim, out_dim])
		fcB2 = bias_variable("fcB2", [out_dim])
		fc2 = fc(fc1, fcW2, fcB2, keepProb)
		fc2_sigmoid = tf.nn.sigmoid(fc2)
		#=======================

		return fc2, fc2_sigmoid
#######################

#######################
# C Network
# 
# reuse=Trueで再利用できる（tf.variable_scope() は，変数の管理に用いるスコープ定義）
def CNet(x, out_dim=1, reuse=False, keepProb=1.0, training=False):
	with tf.variable_scope('CNet') as scope:
		if reuse:
			scope.reuse_variables()
	
		# padding='SAME'のとき、出力のサイズO = 入力サイズI/ストライドS
		# 28/2 = 14
		#convW1 = weight_variable("convW1", [3, 3, 1, 32])
		convW1 = weight_variable("convW1", [3, 3, 1, 4])
		#convB1 = bias_variable("convB1", [32])
		convB1 = bias_variable("convB1", [4])

		conv1 = conv2d_relu(x, convW1, convB1, stride=[1,2,2,1])
		conv1 = batch_norm(conv1, training)
		
		# 14/2 = 7
		#convW2 = weight_variable("convW2", [3, 3, 32, 32])
		convW2 = weight_variable("convW2", [3, 3, 4, 16])
		#convB2 = bias_variable("convB2", [32])
		convB2 = bias_variable("convB2", [16])
		
		conv2 = conv2d_relu(conv1, convW2, convB2, stride=[1,2,2,1])
		conv2 = batch_norm(conv2, training)

		#=======================
		# 特徴マップをembeddingベクトルに変換
		# 2次元画像を１次元に変更して全結合層へ渡す
		# np.prod で配列要素の積を算出
		conv2size = np.prod(conv2.get_shape().as_list()[1:])
		conv2 = tf.reshape(conv2, [-1, conv2size])
		
		# 7 x 7 x 16 -> 128
		hidden_dim = 128
		fcW1 = weight_variable("fcW1", [conv2size, hidden_dim])
		fcB1 = bias_variable("fcB1", [hidden_dim])
		fc1 = fc_relu(conv2, fcW1, fcB1, keepProb)

		# 100 -> out_dim
		fcW2 = weight_variable("fcW2", [hidden_dim, out_dim])
		fcB2 = bias_variable("fcB2", [out_dim])
		fc2 = fc(fc1, fcW2, fcB2, keepProb)
		fc2_sigmoid = tf.nn.sigmoid(fc2)
		#=======================

		return fc2, fc2_sigmoid
#######################

#######################
# Adversarial Network
# 
# reuse=Trueで再利用できる（tf.variable_scope() は，変数の管理に用いるスコープ定義）
def ANet(x, z_dim=1, reuse=False, keepProb=1.0, augRatio=1, training=False):
	with tf.variable_scope('ANet') as scope:
		if reuse:
			scope.reuse_variables()

		h_dim = int(z_dim/2)	
		fcW1 = weight_variable("fcW1", [z_dim, h_dim])
		fcB1 = bias_variable("fcB1", [h_dim])
		fc1 = fc_relu(x, fcW1, fcB1, keepProb)
		fcW2 = weight_variable("fcW2", [h_dim, z_dim])
		fcB2 = bias_variable("fcB2", [z_dim])
		fc2 = fc(fc1, fcW2, fcB2, keepProb)

		mean, var = tf.nn.moments(x, axes=[0])
		sigmaD1 = tf.get_variable('sigmaD1', [z_dim], initializer=tf.constant_initializer(1.0))
		gauss1 = tfd.MultivariateNormalDiag(loc=np.zeros(z_dim,dtype=np.float32), scale_diag=var*sigmaD1)
		noise1 = gauss1.sample(batchSize*augRatio)

		output = tf.tile(fc2,[augRatio,1]) + noise1

		return output
#######################

#######################
# Rのエンコーダとデコーダの連結
xTrain = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
xTrainNoise = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
xTest = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
xTestNoise = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])


# 学習用
encoderR_train = encoderR(xTrainNoise, z_dim_R, keepProb=keepProbTrain, training=True)
decoderR_train = decoderR(encoderR_train, z_dim_R, keepProb=keepProbTrain, training=True)

encoderR_train_abnormal = ANet(encoderR_train, z_dim_R, keepProb=keepProbTrain, augRatio=augRatio, training=True)
decoderR_train_abnormal = decoderR(encoderR_train_abnormal, z_dim_R, reuse=True, keepProb=1.0, training=False)

# テスト用
encoderR_test = encoderR(xTestNoise, z_dim_R, reuse=True, keepProb=1.0)
decoderR_test = decoderR(encoderR_test, z_dim_R, reuse=True, keepProb=1.0)
#######################

#######################
# 学習用

predictFake_logit_train, predictFake_train  = DNet(decoderR_train, keepProb=keepProbTrain, training=True)
predictTrue_logit_train, predictTrue_train = DNet(xTrain,reuse=True, keepProb=keepProbTrain, training=True)

predictNormal_logit_train, predictNormal_train = CNet(xTrain, keepProb=keepProbTrain, training=True)
predictAbnormal_logit_train, predictAbnormal_train = CNet(decoderR_train_abnormal, reuse=True, keepProb=keepProbTrain, training=True)
#######################

#######################
# 損失関数の設定


#====================
# R networks
lossR = tf.reduce_mean(tf.square(decoderR_train - xTrain))
lossRAll = -tf.reduce_mean(tf.log(1 - predictFake_train + lambdaSmall)) + alpha * lossR
'''
lossR = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=decoderR_train, labels=xTrain))
lossRAll = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predictFake_logit_train, labels=tf.zeros_like(predictFake_logit_train))) + alpha * lossR
'''
#====================

#====================
# D and C Networks
lossD = -tf.reduce_mean(tf.log(predictTrue_train  + lambdaSmall)) - tf.reduce_mean(tf.log(1 - predictFake_train +  lambdaSmall))
lossC = -tf.reduce_mean(tf.log(predictAbnormal_train + lambdaSmall)) - tf.reduce_mean(tf.log(1 - predictNormal_train + lambdaSmall))
'''
lossDTrue = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predictTrue_logit_train, labels=tf.ones_like(predictTrue_logit_train))) 
lossDFake= tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predictFake_logit_train, labels=tf.zeros_like(predictFake_logit_train))) 
lossD = lossDTrue + lossDFake

lossCAbnormal = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predictAbnormal_logit_train, labels=tf.ones_like(predictAbnormal_logit_train))) 
lossCNormal = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predictNormal_logit_train, labels=tf.zeros_like(predictNormal_logit_train))) 
lossC = lossCAbnormal + lossCNormal
'''
#====================

#====================
# A Network
decoderSize = np.prod(decoderR_train.get_shape().as_list()[1:])
diff = tf.norm(tf.reshape(decoderR_train_abnormal,[-1,decoderSize]) - tf.tile(tf.reshape(decoderR_train,[-1,decoderSize]),[augRatio,1]), axis=1)/decoderSize
mean, var = tf.nn.moments(tf.reshape(decoderR_train_abnormal,[-1,decoderSize]),axes=[0])
#lossA = tf.reduce_mean(predictAbnormal_train) + tf.reduce_mean(tf.exp(-diff)) + tf.reduce_mean(tf.exp(-var))
lossA = tf.reduce_mean(predictAbnormal_train) + tf.reduce_mean(tf.exp(-diff)) + tf.reduce_mean(tf.exp(-var))
#====================
#######################


#######################
# Update
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="encoderR") + tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="decoderR")
with tf.control_dependencies(extra_update_ops):
	Rvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="encoderR") + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="decoderR")
	trainerR = tf.train.AdamOptimizer(1e-3).minimize(lossR, var_list=Rvars)
	trainerRAll = tf.train.AdamOptimizer(1e-3).minimize(lossRAll, var_list=Rvars)

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="DNet")
with tf.control_dependencies(extra_update_ops):
	Dvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="DNet")
	trainerD = tf.train.AdamOptimizer(1e-3).minimize(lossD, var_list=Dvars)

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="CNet")
with tf.control_dependencies(extra_update_ops):
	Cvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="CNet")
	trainerC = tf.train.AdamOptimizer(1e-3).minimize(lossC, var_list=Cvars)

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="ANet")
with tf.control_dependencies(extra_update_ops):
	Avars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="ANet")
	trainerA = tf.train.AdamOptimizer(1e-3).minimize(lossA, var_list=Avars)

'''
optimizer = tf.train.AdamOptimizer()

# 勾配のクリッピング
gvsR = optimizer.compute_gradients(lossR, var_list=Rvars)
gvsRAll = optimizer.compute_gradients(lossRAll, var_list=Rvars)
gvsD = optimizer.compute_gradients(-lossD, var_list=Dvars)
gvsC = optimizer.compute_gradients(-lossC, var_list=Cvars)

capped_gvsR = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvsR if grad is not None]
capped_gvsRAll = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvsRAll if grad is not None]
capped_gvsD = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvsD if grad is not None]
capped_gvsC = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvsC if grad is not None]

trainerR = optimizer.apply_gradients(capped_gvsR)
trainerRAll = optimizer.apply_gradients(capped_gvsRAll)
trainerD = optimizer.apply_gradients(capped_gvsD)
trainerC = optimizer.apply_gradients(capped_gvsC)
'''
#######################

#######################
#テスト用
predictDX_logit, predictDX = DNet(xTest,reuse=True, keepProb=1.0)
predictDRX_logit, predictDRX = DNet(decoderR_test,reuse=True, keepProb=1.0)
predictCX_logit, predictCX = CNet(xTest,reuse=True, keepProb=1.0)
predictCRX_logit, predictCRX = CNet(decoderR_test,reuse=True, keepProb=1.0)
#######################

#######################
#######################
# メイン
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# ランダムシードの設定
tf.set_random_seed(0)

#=======================
# MNISTのデータの取得
#myData = input_data.read_data_sets("MNIST/",dtype=tf.uint8)
myData = input_data.read_data_sets("MNIST/",dtype=tf.float32)

targetTrainInds = np.where(myData.train.labels == targetChar)[0]
targetTrainData = myData.train.images[myData.train.labels == targetChar]
batchNum = len(targetTrainInds)//batchSize
#=======================

#=======================
# テストデータの準備
normalTestInds = np.where(myData.test.labels == targetChar)[0]

# Trueのindex（シャッフル）
normalTestIndsShuffle = normalTestInds[np.random.permutation(len(normalTestInds))]

# Fakeのindex
abnormalTestInds = np.setdiff1d(np.arange(len(myData.test.labels)),normalTestInds)
#=======================

#=======================
# 評価値、損失を格納するリスト
recallDXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
precisionDXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
f1DXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
aucDXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
aucDXs_inv = [[] for tmp in np.arange(len(testAbnormalRatios))]

recallDRXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
precisionDRXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
f1DRXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
aucDRXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
aucDRXs_inv = [[] for tmp in np.arange(len(testAbnormalRatios))]

recallGXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
precisionGXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
f1GXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
aucGXs = [[] for tmp in np.arange(len(testAbnormalRatios))]

recallCXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
precisionCXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
f1CXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
aucCXs = [[] for tmp in np.arange(len(testAbnormalRatios))]

recallCRXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
precisionCRXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
f1CRXs = [[] for tmp in np.arange(len(testAbnormalRatios))]
aucCRXs = [[] for tmp in np.arange(len(testAbnormalRatios))]

lossR_values = []
lossRAll_values = []
lossD_values = []
lossC_values = []
lossA_values = []
#=======================


batchInd = 0
ite = 0
while not isStop:

	ite = ite + 1	
	#=======================
	# 学習データの作成
	if batchInd == batchNum-1:
		batchInd = 0

	batch = targetTrainData[batchInd*batchSize:(batchInd+1)*batchSize]
	batch_x = np.reshape(batch,(batchSize,28,28,1))

	batchInd += 1
	
	# ノイズを追加する(ガウシアンノイズ)
	batch_x_noise = batch_x + np.random.normal(0,noiseSigma,batch_x.shape)
	
	#batch_x[batch_x < 0] = 0
	#batch_x[batch_x > 255] = 255
	#batch_x[batch_x > 1] = 1
	#=======================

	#=======================
	# ALOCC(Adversarially Learned One-Class Classifier)の学習
	if (trainMode == ALOCC and isTrain):

		# training R network with batch_x & batch_x_noise
		_, lossR_value, lossRAll_value, decoderR_train_value, encoderR_train_value = sess.run(
									[trainerRAll, lossR, lossRAll, decoderR_train, encoderR_train],
									feed_dict={xTrain: batch_x, xTrainNoise: batch_x_noise})

		# training D network with batch_x & batch_x_noise
		_, lossD_value, predictFake_train_value, predictTrue_train_value = sess.run(
									[trainerD, lossD, predictFake_train, predictTrue_train],
									feed_dict={xTrain: batch_x,xTrainNoise: batch_x_noise})

		# Re-training R network with batch_x & batch_x_noise
		_, lossR_value, lossRAll_value, decoderR_train_value, encoderR_train_value = sess.run(
									[trainerRAll, lossR, lossRAll, decoderR_train, encoderR_train],
									feed_dict={xTrain: batch_x, xTrainNoise: batch_x_noise})

	#=======================
	# GAN(Generative Adversarial Net)の学習
	elif (trainMode == GAN):

		# training R network with batch_x & batch_x_noise
		_, lossR_value, lossRAll_value, decoderR_train_value, encoderR_train_value = sess.run(
									[trainerRAll, lossR, lossRAll, decoderR_train, encoderR_train],
									feed_dict={xTrain: batch_x, xTrainNoise: batch_x_noise})

		# training D network with batch_x & batch_x_noise
		_, lossD_value, predictFake_train_value, predictTrue_train_value = sess.run(
									[trainerD, lossD, predictFake_train, predictTrue_train],
									feed_dict={xTrain: batch_x,xTrainNoise: batch_x_noise})

		# Re-training R network with batch_x & batch_x_noise
		_, lossR_value, lossRAll_value, decoderR_train_value, encoderR_train_value = sess.run(
									[trainerRAll, lossR, lossRAll, decoderR_train, encoderR_train],
									feed_dict={xTrain: batch_x, xTrainNoise: batch_x_noise})
	#=======================

	#=======================
	# TRIPLEの学習
	elif (trainMode == TRIPLE):

		# training R network with batch_x & batch_x_noise
		_, lossR_value, lossRAll_value, decoderR_train_value, encoderR_train_value = sess.run(
									[trainerRAll, lossR, lossRAll, decoderR_train, encoderR_train],
									feed_dict={xTrain: batch_x, xTrainNoise: batch_x_noise})

		# training D network with batch_x & batch_x_noise
		_, lossD_value, predictFake_train_value, predictTrue_train_value = sess.run(
									[trainerD, lossD, predictFake_train, predictTrue_train],
									feed_dict={xTrain: batch_x, xTrainNoise: batch_x_noise})

		# training C network 
		_, lossC_value, predictAbnormal_train_value, predictNormal_train_value, decoderR_train_abnormal_value = sess.run(
									[trainerC, lossC, predictAbnormal_train, predictNormal_train, decoderR_train_abnormal],
									feed_dict={xTrain: batch_x, xTrainNoise: batch_x_noise})

		# training A network
		_, lossA_value, diff_value,var_value = sess.run([trainerA, lossA, diff,var],feed_dict={xTrain: batch_x, xTrainNoise: batch_x_noise})

		# Re-training R with batch_x
		_, lossR_value, lossRAll_value, decoderR_train_value, encoderR_train_value = sess.run(
										[trainerRAll, lossR, lossRAll, decoderR_train, encoderR_train],
										feed_dict={xTrain: batch_x, xTrainNoise: batch_x_noise}) 
	#=======================

	#=======================
	# もし誤差が下がらない場合は終了
	if lossR_value < stopTrainThre:
		isTrain = False
	#=======================


	#=======================
	# max iteration 
	if ite >= nIte:
		isStop = True
	#=======================

	#====================
	# 損失の記録
	lossR_values.append(lossR_value)
	lossRAll_values.append(lossRAll_value)
	lossD_values.append(lossD_value)	

	if (trainMode == TRIPLE):
		lossC_values.append(lossC_value)
		lossA_values.append(lossA_value)
	
	if ite%100 == 0:
		if (trainMode == TRIPLE):
			print("%s: #%d %d(%d), lossR=%f, lossRAll=%f, lossD=%f, lossC=%f, lossA=%f" % (trainModeStr, ite, targetChar, trialNo, lossR_value, lossRAll_value, lossD_value, lossC_value, lossA_value))
		else:
			print("%s: #%d %d(%d), lossR=%f, lossRAll=%f, lossD=%f" % (trainModeStr, ite, targetChar, trialNo, lossR_value, lossRAll_value, lossD_value))
	#====================

	#######################
	# Evaluation
	if (ite % 1000 == 0):
		
		#====================
		# training data
		if isVisualize:
			plt.close()

			# plot example of true, fake, reconstructed x

			plt.imshow(batch_x[0,:,:,0],cmap="gray")
			plt.savefig(visualPath+"x_true.eps")

			plt.imshow(batch_x_noise[0,:,:,0],cmap="gray")
			plt.savefig(visualPath+"x_fake.eps")
		
			plt.imshow(decoderR_train_value[0,:,:,0],cmap="gray")
			plt.savefig(visualPath+"x_reconstructed.eps")

			if trainMode == TRIPLE:	
				for i in np.arange(10):
					plt.imshow(decoderR_train_abnormal_value[i,:,:,0],cmap="gray")
					plt.savefig(visualPath+"x_aug_{}.eps".format(i))
		#====================
		
		#====================
		# テストデータ	

		#--------------------------
		# variables to keep values
		predictDX_value = [[] for tmp in np.arange(len(testAbnormalRatios))]
		predictDRX_value = [[] for tmp in np.arange(len(testAbnormalRatios))]
		decoderR_test_value = [[] for tmp in np.arange(len(testAbnormalRatios))]
		encoderR_test_value = [[] for tmp in np.arange(len(testAbnormalRatios))]
		
		if (trainMode == GAN):
			print("min:{}, max:{}".format(np.min(predictAbnormal_train_value),np.max(predictAbnormal_train_value)))
			predictCX_value = [[] for tmp in np.arange(len(testAbnormalRatios))]

		if (trainMode == TRIPLE):
			print("min:{}, max:{}".format(np.min(predictAbnormal_train_value),np.max(predictAbnormal_train_value)))
			predictCX_value = [[] for tmp in np.arange(len(testAbnormalRatios))]
			predictCRX_value = [[] for tmp in np.arange(len(testAbnormalRatios))]
		#--------------------------

		#--------------------------
		# loop for anomaly ratios
		for ind, testAbnormalRatio in enumerate(testAbnormalRatios):

			# データの数
			abnormalNum = int(np.floor(len(normalTestInds)*testAbnormalRatio))
			normalNum = len(normalTestInds) - abnormalNum
			
			# Trueのindex
			normalTestIndsSelected = normalTestInds[:normalNum]

			# Fakeのindex
			abnormalTestIndsSelected = abnormalTestInds[:abnormalNum]

			# reshape & concat
			test_x = np.reshape(myData.test.images[normalTestIndsSelected],(len(normalTestIndsSelected),28,28,1))
			test_x_fake = np.reshape(myData.test.images[abnormalTestIndsSelected],(len(abnormalTestIndsSelected),28,28,1))
			test_x = np.vstack([test_x_fake, test_x])

			# add noise
			test_x_noise = test_x + np.random.normal(0,noiseSigma,test_x.shape)
			test_y = np.hstack([np.ones(len(abnormalTestIndsSelected)),np.zeros(len(normalTestIndsSelected))])
			test_y_inv = np.hstack([np.zeros(len(abnormalTestIndsSelected)),np.ones(len(normalTestIndsSelected))])

			#--------------------------
			if trainMode == ALOCC:
				predictDX_value[ind], predictDRX_value[ind], decoderR_test_value[ind], encoderR_test_value[ind] = sess.run(
								[predictDX, predictDRX, decoderR_test, encoderR_test],
								feed_dict={xTest: test_x, xTestNoise: test_x})
								
			elif trainMode == GAN:
				predictDX_value[ind], predictDRX_value[ind], decoderR_test_value[ind], encoderR_test_value[ind] = sess.run(
								[predictDX, predictDRX, decoderR_test, encoderR_test],
								feed_dict={xTest: test_x, xTestNoise: test_x})
								
				# difference between original and recovered data
				dataSize = np.prod(test_x.shape()[1:])
				predictGX_value = np.square(np.reshape(test_x,[-1,dataSize]) - np.reshape(decoderR_test_value[-1,dataSize]))
				predictGX_value = dataSize - predictGX_value
				
				pdb.set_trace()
								
								
			elif trainMode == TRIPLE:
				predictDX_value[ind], predictDRX_value[ind], decoderR_test_value[ind], encoderR_test_value[ind] = sess.run(
								[predictDX, predictDRX, decoderR_test, encoderR_test],
								feed_dict={xTest: test_x, xTestNoise: test_x_noise})

				predictCX_value[ind], predictCRX_value[ind] = sess.run([predictCX, predictCRX], feed_dict={xTest: test_x, xTestNoise: test_x_noise})
			#--------------------------

			#--------------------------
			# 評価値の計算と記録 D Network
			recallDX, precisionDX, f1DX, aucDX = calcEval(1-predictDX_value[ind][:,0], test_y, threAbnormal)
			recallDRX, precisionDRX, f1DRX, aucDRX = calcEval(1-predictDRX_value[ind][:,0], test_y, threAbnormal)


			recallDX_inv, precisionDX_inv, f1DX_inv, aucDX_inv = calcEval(predictDX_value[ind][:,0], test_y_inv, threAbnormal)
			recallDRX_inv, precisionDRX_inv, f1DRX_inv, aucDRX_inv = calcEval(predictDRX_value[ind][:,0], test_y_inv, threAbnormal)

			recallDXs[ind].append(recallDX)
			precisionDXs[ind].append(precisionDX)
			f1DXs[ind].append(f1DX)
			aucDXs[ind].append(aucDX)
			aucDXs_inv[ind].append(aucDX_inv)
		
			recallDRXs[ind].append(recallDRX)
			precisionDRXs[ind].append(precisionDRX)
			f1DRXs[ind].append(f1DRX)
			aucDRXs[ind].append(aucDRX)
			aucDRXs_inv[ind].append(aucDRX_inv)
			#--------------------------

			#--------------------------
			# GAN
			if trainMode == GAN:
				recallGX, precisionGX, f1GX, aucGX = calcEval(predictGX_value[ind][:,0], test_y, threAbnormal)

				recallGXs[ind].append(recallGX)
				precisionGXs[ind].append(precisionGX)
				f1GXs[ind].append(f1GX)
				aucGXs[ind].append(aucGX)
		
				recallGRXs[ind].append(recallGRX)
				precisionGRXs[ind].append(precisionGRX)
				f1GRXs[ind].append(f1GRX)
				aucGRXs[ind].append(aucGRX)
			#--------------------------

			#--------------------------
			# C Network
			if trainMode == TRIPLE:
				recallCX, precisionCX, f1CX, aucCX = calcEval(predictCX_value[ind][:,0], test_y, threAbnormal)
				recallCRX, precisionCRX, f1CRX, aucCRX = calcEval(predictCRX_value[ind][:,0], test_y, threAbnormal)

				recallCXs[ind].append(recallCX)
				precisionCXs[ind].append(precisionCX)
				f1CXs[ind].append(f1CX)
				aucCXs[ind].append(aucCX)
		
				recallCRXs[ind].append(recallCRX)
				precisionCRXs[ind].append(precisionCRX)
				f1CRXs[ind].append(f1CRX)
				aucCRXs[ind].append(aucCRX)
			#--------------------------

			#--------------------------
			print("ratio:%f" % (testAbnormalRatio))
			print("recallDX=%f, precisionDX=%f, f1DX=%f, aucDX=%f, aucDX_inv=%f" % (recallDX, precisionDX, f1DX, aucDX, aucDX_inv))
			print("recallDRX=%f, precisionDRX=%f, f1DRX=%f, aucDRX=%f, aucDRX_inv=%f" % (recallDRX, precisionDRX, f1DRX, aucDRX, aucDRX_inv))
			
			if trainMode == GAN:
				print("recallGX=%f, precisionGX=%f, f1GX=%f, aucGX=%f" % (recallGX, precisionGX, f1GX, aucGX))

			if trainMode == TRIPLE:
				print("recallCX=%f, precisionCX=%f, f1CX=%f, aucCX=%f" % (recallCX, precisionCX, f1CX, aucCX))
				print("recallCRX=%f, precisionCRX=%f, f1CRX=%f, aucCRX=%f" % (recallCRX, precisionCRX, f1CRX, aucCRX))
			#--------------------------

			if ind == 0:
				#--------------------------
				# 学習で用いている画像（元の画像、ノイズ付加した画像、decoderで復元した画像）を保存
				plt.close()
				fig, figInds = plt.subplots(nrows=3, ncols=nPlotImg, sharex=True)
	
				for figInd in np.arange(figInds.shape[1]):
					fig0 = figInds[0][figInd].imshow(batch_x[figInd,:,:,0],cmap="gray")
					fig1 = figInds[1][figInd].imshow(batch_x_noise[figInd,:,:,0],cmap="gray")
					fig2 = figInds[2][figInd].imshow(decoderR_train_value[figInd,:,:,0],cmap="gray")

					# ticks, axisを隠す
					fig0.axes.get_xaxis().set_visible(False)
					fig0.axes.get_yaxis().set_visible(False)
					fig0.axes.get_xaxis().set_ticks([])
					fig0.axes.get_yaxis().set_ticks([])
					fig1.axes.get_xaxis().set_visible(False)
					fig1.axes.get_yaxis().set_visible(False)
					fig1.axes.get_xaxis().set_ticks([])
					fig1.axes.get_yaxis().set_ticks([])
					fig2.axes.get_xaxis().set_visible(False)
					fig2.axes.get_yaxis().set_visible(False)
					fig2.axes.get_xaxis().set_ticks([])
					fig2.axes.get_yaxis().set_ticks([])					

				path = os.path.join(visualPath,"img_train{}_{}_{}.png".format(postFix,testAbnormalRatio,ite))
				plt.savefig(path)
				#--------------------------

				#--------------------------
				# 提案法で生成した画像（元の画像、提案法で生成たい異常画像）を保存
				if isEmbedSampling & (trainMode > ALOCC):
					path = os.path.join(visualPath,"img_train_aug{}_{}_{}.png".format(postFix,testAbnormalRatio,ite))
					plotImg(batch_x[:nPlotImg], decoderR_train_abnormal_value[:nPlotImg],path)
				#--------------------------
							
				#--------------------------
				# 評価画像のうち正常のものを保存
				path = os.path.join(visualPath,"img_test_true{}_{}_{}.png".format(postFix,testAbnormalRatio,ite))
				plotImg(test_x[-nPlotImg:], decoderR_test_value[ind][-nPlotImg:],path)
				#--------------------------
		
				#--------------------------
				# 評価画像のうち異常のものを保存
				path = os.path.join(visualPath,"img_test_fake{}_{}_{}.png".format(postFix,testAbnormalRatio,ite))
				plotImg(test_x[:nPlotImg], decoderR_test_value[ind][:nPlotImg],path)
				#--------------------------
		
			#--------------------------
			# Visualizing embedded space z
			if isVisualize & (z_dim_R == 2):
				plt.close()

				#---------------
				# train data

				# get z samples for true 
				encoderR_train_value, encoderR_train_abnormal_value = sess.run([encoderR_train,encoderR_train_abnormal], feed_dict={xTrain: batch_x, xTrainNoise:batch_x_noise})
				# plot example of embedded vectors, z
				plt.plot(encoderR_train_value[:,0],encoderR_train_value[:,1],'o',markersize=6,markeredgecolor="b",markerfacecolor="w")
				plt.plot(encoderR_train_value[:,0],encoderR_train_value[:,1],'d',markersize=6,markeredgecolor="g",markerfacecolor="w")

				if trainMode == TRIPLE:
					plt.plot(encoderR_train_abnormal_value[:,0],encoderR_train_abnormal_value[:,1],'rd',markersize=6)

					plt.xlim(int(np.min([np.min(encoderR_train_abnormal_value[:,0]), np.min(encoderR_train_value[:,0]),np.min(encoderR_train_value[:,0])]) - 100),
						int(np.max([np.max(encoderR_train_abnormal_value[:,0]), np.max(encoderR_train_value[:,0]),np.max(encoderR_train_value[:,0])]) + 100) )

					plt.ylim(int(np.min([np.min(encoderR_train_abnormal_value[:,1]), np.min(encoderR_train_value[:,1]),np.min(encoderR_train_value[:,1])]) - 100),
						int(np.max([np.max(encoderR_train_abnormal_value[:,1]), np.max(encoderR_train_value[:,1]),np.max(encoderR_train_value[:,1])]) + 100) )

				else:
					plt.xlim(int(np.min([np.min(encoderR_train_value[:,0]),np.min(encoderR_train_value[:,0])]) - 100),
						int(np.max([np.max(encoderR_train_value[:,0]),np.max(encoderR_train_value[:,0])]) + 100) )

					plt.ylim(int(np.min([np.min(encoderR_train_value[:,1]),np.min(encoderR_train_value[:,1])]) - 100),
						int(np.max([np.max(encoderR_train_value[:,1]),np.max(encoderR_train_value[:,1])]) + 100) )


				plt.savefig(visualPath+"z.eps")
				#---------------

				#---------------
				# test data

				# plot example of embedded vectors, z
				plt.plot(encoderR_test_value[ind][:normalNum,0],encoderR_test_value[ind][:normalNum,1],'o',markersize=3,markeredgecolor="g",markerfacecolor="None")
				plt.plot(encoderR_test_value[ind][normalNum:,0],encoderR_test_value[ind][normalNum:,1],'o',markersize=3,markeredgecolor="r",markerfacecolor="None")
				
				if trainMode == TRIPLE:
					plt.plot(encoderR_train_abnormal_value[:,0],encoderR_train_abnormal_value[:,1],'.',markersize=2,markeredgecolor="m",markerfacecolor="None")
					plt.xlim(int(np.min([np.min(encoderR_train_abnormal_value[:,0]), np.min(encoderR_test_value[ind][:,0]),np.min(encoderR_test_value[ind][:,0])]) - 100),
						int(np.max([np.max(encoderR_train_abnormal_value[:,0]), np.max(encoderR_test_value[ind][:,0]),np.max(encoderR_test_value[ind][:,0])]) + 100) )

					plt.ylim(int(np.min([np.min(encoderR_train_abnormal_value[:,1]), np.min(encoderR_test_value[ind][:,1]),np.min(encoderR_test_value[ind][:,1])]) - 100),
						int(np.max([np.max(encoderR_train_abnormal_value[:,1]), np.max(encoderR_test_value[ind][:,1]),np.max(encoderR_test_value[ind][:,1])]) + 100) )

				else:
					plt.xlim(int(np.min([np.min(encoderR_test_value[ind][:,0]),np.min(encoderR_test_value[ind][:,0])]) - 100),
						int(np.max([np.max(encoderR_test_value[ind][:,0]),np.max(encoderR_test_value[ind][:,0])]) + 100) )

					plt.ylim(int(np.min([np.min(encoderR_test_value[ind][:,1]),np.min(encoderR_test_value[ind][:,1])]) - 100),
						int(np.max([np.max(encoderR_test_value[ind][:,1]),np.max(encoderR_test_value[ind][:,1])]) + 100) )

				plt.savefig(visualPath+"z_test_{}_{}.eps".format(ite,ind))
			#--------------------------
		#====================

		#=======================
		# チェックポイントの保存
		saver = tf.train.Saver()
		saver.save(sess,modelPath+"model{}.ckpt".format(postFix))
		#=======================

	#######################
		
#######################
# pickleに保存
path = os.path.join(logPath,"log{}.pickle".format(postFix))
with open(path, "wb") as fp:
	pickle.dump(batch_x,fp)
	pickle.dump(batch_x_noise,fp)
	pickle.dump(encoderR_train_value,fp)
	pickle.dump(decoderR_train_value,fp)
	pickle.dump(predictFake_train_value,fp)
	pickle.dump(predictTrue_train_value,fp)	
	pickle.dump(test_x,fp)
	pickle.dump(test_y,fp)
	pickle.dump(decoderR_test_value,fp)
	pickle.dump(predictDX_value,fp)
	pickle.dump(predictDRX_value,fp)
	pickle.dump(recallDXs,fp)
	pickle.dump(precisionDXs,fp)
	pickle.dump(f1DXs,fp)
	pickle.dump(aucDXs,fp)
	pickle.dump(aucDXs_inv,fp)
	pickle.dump(recallDRXs,fp)
	pickle.dump(precisionDRXs,fp)
	pickle.dump(f1DRXs,fp)
	pickle.dump(aucDRXs,fp)
	pickle.dump(aucDRXs_inv,fp)

	if trainMode == TRIPLE:
		pickle.dump(recallCXs,fp)
		pickle.dump(precisionCXs,fp)
		pickle.dump(f1CXs,fp)
		pickle.dump(aucCXs,fp)
		pickle.dump(recallCRXs,fp)
		pickle.dump(precisionCRXs,fp)
		pickle.dump(f1CRXs,fp)
		pickle.dump(aucCRXs,fp)	

	pickle.dump(lossR_values,fp)
	pickle.dump(lossRAll_values,fp)
	pickle.dump(lossD_values,fp)

	if trainMode == TRIPLE:
		pickle.dump(lossC_values,fp)
		pickle.dump(lossA_values,fp)
		pickle.dump(decoderR_train_abnormal_value,fp)

	pickle.dump(params,fp)
#######################

#######################
#######################
