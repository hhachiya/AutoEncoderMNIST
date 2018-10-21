# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.ops import nn_ops
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.cluster import KMeans
import numpy as np
import math, os
import pickle
import pdb
#import input_data
import matplotlib.pylab as plt
import sys


#===========================
# ランダムシード
np.random.seed(0)
#===========================

#===========================
# パラメータの設定
z_dim_R = 100
#z_dim_R = 2

# 学習モード
ALOCC = 0
ALDAD = 1
isStop = False
isEmbedSampling = False
isTrainingD = False

if len(sys.argv) > 1:
	# 文字の種類
	trainMode = int(sys.argv[1])

	# trail no.
	if len(sys.argv) > 2:
		targetChar = int(sys.argv[2])
		trialNo = int(sys.argv[3])
		noiseSigma = int(sys.argv[4])
	else:
		targetChar = 0
		trialNo = 0	
		noiseSigma = 40

else:
	# 方式の種類
	trainMode = ALDAD

# Rの二乗誤差の重み係数
lambdaR = 0.4

# log(0)と0割防止用
lambdaSmall = 10e-8
#lambdaSmall = 0.00001

# テストデータにおける偽物の割合
testFakeRatios = [0.1, 0.2, 0.3, 0.4, 0.5]

# 予測結果に対する閾値
threFake = 0.5

# Rの誤差の閾値
threLossR = 50

# Dの誤差の閾値
threLossD = -10e-8

# データ拡張のパラメータ
clusterNum = 5
augNum = 100

# バッチデータ数
batchSize = 300

# 変数をまとめたディクショナリ
params = {'z_dim_R':z_dim_R, 'testFakeRatios':testFakeRatios, 'labmdaR':lambdaR,
'threFake':threFake, 'targetChar':targetChar,'batchSize':batchSize}

# ノイズの大きさ
#noiseSigma = 0.155
#noiseSigma = 100
#noiseSigma = 40

# embedded spaceにおけるノイズのレベル
noiseSigmaEmbed = 3

# プロットする画像数
nPlotImg = 10

# ファイル名のpostFix
if trainMode == ALOCC:
	postFix = "_ALOCC_{}_{}_{}_{}".format(targetChar, trialNo, z_dim_R, noiseSigma)
elif trainMode == ALDAD:
	postFix = "_ALDAD_{}_{}_{}_{}_{}".format(targetChar, trialNo, z_dim_R, noiseSigma, noiseSigmaEmbed)

# 反復回数
nIte = 5000

visualPath = 'visualization'
modelPath = 'models'
logPath = 'logs'
#===========================

#===========================
# 評価値の計算用の関数
def calcEval(predict, gt, threFake=0.5):
	predict[predict >= threFake] = 1.
	predict[predict < threFake] = 0.

	recall = np.sum(predict[gt==1])/np.sum(gt==1)
	precision = np.sum(predict[gt==1])/np.sum(predict==1)
	f1 = 2 * (precision * recall)/(precision + recall)

	return recall, precision, f1
#===========================

#===========================
# レイヤーの関数
def weight_variable(name, shape):
	return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(stddev=0.1))
	
def bias_variable(name, shape):
	return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.1))

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
def conv2d_t_relu(inputs, w, b, output_shape, stride):
	conv = tf.nn.conv2d_transpose(inputs, w, output_shape=output_shape, strides=stride, padding='SAME') + b
	conv = tf.nn.relu(conv)
	return conv

# 2D deconvolution layer
def conv2d_t(inputs, w, b, output_shape, stride):
	conv = tf.nn.conv2d_transpose(inputs, w, output_shape=output_shape, strides=stride, padding='SAME') + b
	return conv
	
# fc layer with ReLU
def fc_relu(inputs, w, b, keepProb=1.0):
	fc = tf.matmul(inputs, w) + b
	fc = tf.nn.dropout(fc, keepProb)
	fc = tf.nn.relu(fc)
	return fc
	
# fc layer with softmax
def fc_sigmoid(inputs, w, b, keepProb=1.0):
	fc = tf.matmul(inputs, w) + b
	fc = tf.nn.dropout(fc, keepProb)
	fc = tf.nn.sigmoid(fc)
	return fc
#===========================

#===========================
# エンコーダ
# 画像をz_dim次元のベクトルにエンコード
# reuse=Trueで再利用できる（tf.variable_scope() は，変数の管理に用いるスコープ定義）
def encoderR(x, z_dim, reuse=False, keepProb = 1.0):
	with tf.variable_scope('encoderR') as scope:
		if reuse:
			scope.reuse_variables()
	
		# padding='SAME'のとき、出力のサイズO = 入力サイズI/ストライドS
		# 28/2 = 14
		convW1 = weight_variable("convW1", [3, 3, 1, 32])
		convB1 = bias_variable("convB1", [32])
		conv1 = conv2d_relu(x, convW1, convB1, stride=[1,2,2,1])
		
		# 14/2 = 7
		convW2 = weight_variable("convW2", [3, 3, 32, 32])
		convB2 = bias_variable("convB2", [32])
		conv2 = conv2d_relu(conv1, convW2, convB2, stride=[1,2,2,1])

		#--------------
		# 特徴マップをembeddingベクトルに変換
		# 2次元画像を１次元に変更して全結合層へ渡す
		# np.prod で配列要素の積を算出
		conv2size = np.prod(conv2.get_shape().as_list()[1:])
		conv2 = tf.reshape(conv2, [-1, conv2size])
		
		# 7 x 7 x 32 -> z-dim
		fcW1 = weight_variable("fcW1", [conv2size, z_dim])
		fcB1 = bias_variable("fcB1", [z_dim])
		fc1 = fc_relu(conv2, fcW1, fcB1, keepProb)
		#fc1 = fc_sigmoid(conv2, fcW1, fcB1, keepProb)
		#--------------

		return fc1
#===========================

#===========================
# デコーダ
# z_dim次元の画像にデコード
# reuse=Trueで再利用できる（tf.variable_scope() は，変数の管理に用いるスコープ定義）
def decoderR(z,z_dim,reuse=False, keepProb = 1.0):
	with tf.variable_scope('decoderR') as scope:
		if reuse:
			scope.reuse_variables()

		#--------------
		# embeddingベクトルを特徴マップに変換
		# 2次元画像を１次元に変更して全結合層へ渡す
		fcW1 = weight_variable("fcW1", [z_dim, 7*7*32])
		fcB1 = bias_variable("fcB1", [7*7*32])
		fc1 = fc_relu(z, fcW1, fcB1, keepProb)

		batchSize = tf.shape(fc1)[0]
		fc1 = tf.reshape(fc1, tf.stack([batchSize, 7, 7, 32]))
		#--------------
		
		# padding='SAME'のとき、出力のサイズO = 入力サイズI/ストライドS
		# 7 x 2 = 14
		convW1 = weight_variable("convW1", [3, 3, 32, 32])
		convB1 = bias_variable("convB1", [32])
		conv1 = conv2d_t_relu(fc1, convW1, convB1, output_shape=[batchSize,14,14,32], stride=[1,2,2,1])
		
		# 14 x 2 = 28
		convW2 = weight_variable("convW2", [3, 3, 1, 32])
		convB2 = bias_variable("convB2", [1])
		output = conv2d_t_relu(conv1, convW2, convB2, output_shape=[batchSize,28,28,1], stride=[1,2,2,1])
		#output = conv2d_t_sigmoid(conv1, convW2, convB2, output_shape=[batchSize,28,28,1], stride=[1,2,2,1])
		
		return output
#===========================

#===========================
# D Network
# 
# reuse=Trueで再利用できる（tf.variable_scope() は，変数の管理に用いるスコープ定義）
def DNet(x, z_dim=1, reuse=False, keepProb=1.0):
	with tf.variable_scope('DNet') as scope:
		if reuse:
			scope.reuse_variables()
	
		# padding='SAME'のとき、出力のサイズO = 入力サイズI/ストライドS
		# 28/2 = 14
		convW1 = weight_variable("convW1", [3, 3, 1, 32])
		convB1 = bias_variable("convB1", [32])
		conv1 = conv2d_relu(x, convW1, convB1, stride=[1,2,2,1])
		
		# 14/2 = 7
		convW2 = weight_variable("convW2", [3, 3, 32, 32])
		convB2 = bias_variable("convB2", [32])
		conv2 = conv2d_relu(conv1, convW2, convB2, stride=[1,2,2,1])

		#--------------
		# 特徴マップをembeddingベクトルに変換
		# 2次元画像を１次元に変更して全結合層へ渡す
		# np.prod で配列要素の積を算出
		conv2size = np.prod(conv2.get_shape().as_list()[1:])
		conv2 = tf.reshape(conv2, [-1, conv2size])
		
		# 7 x 7 x 32 -> z-dim
		fcW1 = weight_variable("fcW1", [conv2size, z_dim])
		fcB1 = bias_variable("fcB1", [z_dim])
		fc1 = fc_sigmoid(conv2, fcW1, fcB1, keepProb)
		#--------------

		return fc1
#===========================

#===========================
# Rのエンコーダとデコーダの連結
xTrue = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
xFake = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
xTest = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
encoderR_train_aug = tf.placeholder(tf.float32, shape=[None, z_dim_R])

# 学習用
encoderR_train = encoderR(xFake, z_dim_R, keepProb=1.0)
decoderR_train = decoderR(encoderR_train, z_dim_R, keepProb=1.0)
decoderR_train_aug = decoderR(encoderR_train_aug, z_dim_R, reuse=True, keepProb=1.0)

# テスト用
encoderR_test = encoderR(xTest, z_dim_R, reuse=True, keepProb=1.0)
decoderR_test = decoderR(encoderR_test, z_dim_R, reuse=True, keepProb=1.0)
#===========================

#===========================
# MMD
def compute_kernel(x, y):
	x_size = tf.shape(x)[0]
	y_size = tf.shape(y)[0]
	dim = tf.shape(x)[1]
	tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
	tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
	return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

def compute_mmd(x, y):
	x_kernel = compute_kernel(x, x)
	y_kernel = compute_kernel(y, y)
	xy_kernel = compute_kernel(x, y)
	return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)
#===========================
	
#===========================
# 損失関数の設定

#学習用
predictFake_train = DNet(decoderR_train, keepProb=1.0)
predictFake_train_aug = DNet(decoderR_train_aug, reuse=True, keepProb=1.0)
predictTrue_train = DNet(xTrue,reuse=True, keepProb=1.0)

# random encoder samples
encoderR_sample = tf.random_normal(tf.stack([batchSize, z_dim_R]))
lossMMD = compute_mmd(encoderR_sample, encoderR_train)
	
lossR = tf.reduce_mean(tf.square(decoderR_train - xTrue))
#lossRAll = tf.reduce_mean(tf.log(1 - predictFake_train + lambdaSmall)) + lambdaR * lossR + lossMMD
lossRAll = tf.reduce_mean(tf.log(1 - predictFake_train + lambdaSmall)) + lambdaR * lossR
lossD = tf.reduce_mean(tf.log(predictTrue_train  + lambdaSmall)) + tf.reduce_mean(tf.log(1 - predictFake_train +  lambdaSmall))
lossD_aug = tf.reduce_mean(tf.log(predictTrue_train  + lambdaSmall)) + tf.reduce_mean(tf.log(1 - predictFake_train_aug +  lambdaSmall))

# R & Dの変数
Rvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="encoderR") + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="decoderR")
Dvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="DNet")

#--------------
# ランダムシードの設定
tf.set_random_seed(0)
#--------------

trainerR = tf.train.AdamOptimizer(1e-3).minimize(lossR, var_list=Rvars)
trainerRAll = tf.train.AdamOptimizer(1e-3).minimize(lossRAll, var_list=Rvars)
trainerD = tf.train.AdamOptimizer(1e-3).minimize(-lossD, var_list=Dvars)
trainerD_aug = tf.train.AdamOptimizer(1e-3).minimize(-lossD_aug, var_list=Dvars)

'''
optimizer = tf.train.AdamOptimizer()

# 勾配のクリッピング
gvsR = optimizer.compute_gradients(lossR, var_list=Rvars)
gvsRAll = optimizer.compute_gradients(lossRAll, var_list=Rvars)
gvsD = optimizer.compute_gradients(-lossD, var_list=Dvars)

capped_gvsR = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvsR if grad is not None]
capped_gvsRAll = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvsRAll if grad is not None]
capped_gvsD = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvsD if grad is not None]

trainerR = optimizer.apply_gradients(capped_gvsR)
trainerRAll = optimizer.apply_gradients(capped_gvsRAll)
trainerD = optimizer.apply_gradients(capped_gvsD)
'''
#===========================

#===========================
#テスト用
predictDX = DNet(xTest,reuse=True, keepProb=1.0)
predictDRX = DNet(decoderR_test,reuse=True, keepProb=1.0)
#===========================

#===========================
# メイン
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#--------------
# MNISTのデータの取得
myData = input_data.read_data_sets("MNIST/",dtype=tf.uint8)
#myData = input_data.read_data_sets("MNIST/")

targetTrainInds = np.where(myData.train.labels == targetChar)[0]
targetTrainData = myData.train.images[myData.train.labels == targetChar]
batchNum = len(targetTrainInds)//batchSize
#--------------

#--------------
# テストデータの準備
targetTestInds = np.where(myData.test.labels == targetChar)[0]

# Trueのindex（シャッフル）
targetTestIndsShuffle = targetTestInds[np.random.permutation(len(targetTestInds))]

# Fakeのindex（シャッフル）
fakeTestInds = np.setdiff1d(np.arange(len(myData.test.labels)),targetTestInds)
#--------------

#--------------
# 評価値、損失を格納するリスト
recallDXs = [[] for tmp in np.arange(len(testFakeRatios))]
precisionDXs = [[] for tmp in np.arange(len(testFakeRatios))]
f1DXs = [[] for tmp in np.arange(len(testFakeRatios))]
recallDRXs = [[] for tmp in np.arange(len(testFakeRatios))]
precisionDRXs = [[] for tmp in np.arange(len(testFakeRatios))]
f1DRXs = [[] for tmp in np.arange(len(testFakeRatios))]

lossR_values = []
lossRAll_values = []
lossD_values = []
#--------------


batchInd = 0
ite = 0
#for ite in range(nIte):
while not isStop:

	ite = ite + 1	
	#--------------
	# 学習データの作成
	if batchInd == batchNum-1:
		batchInd = 0

	#batch = myData.train.next_batch(batchSize)
	#batch_x_all = np.reshape(batch[0],(batchSize,28,28,1))

	# targetCharのみのデータ
	#targetTrainInds = np.where(batch[1] == targetChar)[0]
	#batch_x = batch_x_all[targetTrainInds]

	batch = targetTrainData[batchInd*batchSize:(batchInd+1)*batchSize]
	batch_x = np.reshape(batch,(batchSize,28,28,1))

	batchInd += 1
	
	# ノイズを追加する(ガウシアンノイズ)
	# 正規分布に従う乱数を出力
	batch_x_fake = batch_x + np.random.normal(0,noiseSigma,batch_x.shape)
	#--------------

	#==============
	# ALOCC(Adversarially Learned One-Class Classifier)の学習
	if trainMode == ALOCC:

		if ite == 2000:
			isTrainingD = True

		# training R network with batch_x & batch_x_fake
		_, lossR_value, lossRAll_value, decoderR_train_value, encoderR_train_value = sess.run(
											[trainerRAll, lossR, lossRAll, decoderR_train, encoderR_train],
											feed_dict={xTrue: batch_x, xFake: batch_x_fake})

		if isTrainingD:
			# training D network with batch_x & batch_x_fake
			_, lossD_value, predictFake_train_value, predictTrue_train_value = sess.run(
											[trainerD, lossD, predictFake_train, predictTrue_train],
											feed_dict={xTrue: batch_x,xFake: batch_x_fake})

		# Re-training R network with batch_x & batch_x_fake
		#_, lossR_value, lossRAll_value, decoderR_train_value, encoderR_train_value = sess.run(
		#									[trainerRAll, lossR, lossRAll, decoderR_train, encoderR_train],
		#									feed_dict={xTrue: batch_x, xFake: batch_x_fake})


		if lossR_value < threLossR:
			print("stopped training")
			isStop = True
	#==============

	#==============
	# ALDAD(Adversarially Learned Discriminative Abnormal Detector)の学習
	elif trainMode == ALDAD:
		if ite == 2000:
			isEmbedSampling = True

		if ite >= nIte:
			isStop = True


		# training R with batch_x
		_, lossR_value, lossRAll_value, decoderR_train_value, encoderR_train_value = sess.run(
										[trainerRAll, lossR, lossRAll, decoderR_train, encoderR_train],
										feed_dict={xTrue: batch_x, xFake: batch_x_fake})

		'''	
		# training D with batch_x
		if not isEmbedSampling:
			_, lossD_value, predictFake_train_value, predictTrue_train_value = sess.run(
										[trainerD, lossD, predictFake_train, predictTrue_train],
										feed_dict={xTrue: batch_x, xFake: batch_x_fake})
		'''

		lossD_value = -1

		if isEmbedSampling:
			#------------
			# clustering samples in embeded space, z
			kmeans = KMeans(n_clusters=clusterNum, random_state=0).fit(encoderR_train_value)
			means = kmeans.cluster_centers_

			# approximate the probability distribution of z, p_theta(z)
			stds = np.array([np.std(encoderR_train_value[kmeans.labels_==ind,:], axis=0) for ind in np.arange(clusterNum)])

			# sampling from approximated probability distribution, p_theta(z)
			aug_z = np.reshape(np.array([means[ind,:] + np.multiply(np.random.randn(augNum,z_dim_R),
				np.tile(stds[ind,:]*noiseSigmaEmbed,[augNum,1])) for ind in np.arange(clusterNum)]),[-1,z_dim_R])

			#------------

			_, lossD_value, predictFake_train_value, predictTrue_train_value, decoderR_train_aug_value = sess.run([trainerD_aug, lossD_aug, predictFake_train_aug, predictTrue_train, decoderR_train_aug],feed_dict={xTrue: batch_x, encoderR_train_aug: aug_z})

	# もし誤差が下がらない場合はやり直し
	if (ite == 3000) & (lossD_value < -10):
		isTrainedD = False
		isEmbedSampling = False
			
			
	#==============

	# 損失の記録
	lossR_values.append(lossR_value)
	lossRAll_values.append(lossRAll_value)
	lossD_values.append(lossD_value)
	
	if ite%100 == 0:
		print("#%d %d(%d), lossR=%f, lossRAll=%f, lossD=%f" % (ite, targetChar, trialNo, lossR_value, lossRAll_value, lossD_value))
	#--------------

	#--------------
	# テスト
	if (ite % 1000 == 0) | isStop:
	
		
		predictDX_value = [[] for tmp in np.arange(len(testFakeRatios))]
		predictDRX_value = [[] for tmp in np.arange(len(testFakeRatios))]
		decoderR_test_value = [[] for tmp in np.arange(len(testFakeRatios))]
		
		#--------------
		# テストデータの作成	
		for ind, testFakeRatio in enumerate(testFakeRatios):
		
			# データの数
			fakeNum = int(np.floor(len(targetTestInds)*testFakeRatio))
			targetNum = len(targetTestInds) - fakeNum
			
			# Trueのindex
			targetTestIndsSelected = targetTestIndsShuffle[:targetNum]

			# Fakeのindex
			fakeTestIndsSelected = fakeTestInds[np.random.permutation(len(fakeTestInds))[:fakeNum]]

			# reshape & concat
			test_x = np.reshape(myData.test.images[targetTestIndsSelected],(len(targetTestIndsSelected),28,28,1))
			test_x_fake = np.reshape(myData.test.images[fakeTestIndsSelected],(len(fakeTestIndsSelected),28,28,1))
			test_x = np.vstack([test_x, test_x_fake])
			test_y = np.hstack([np.ones(len(targetTestIndsSelected)),np.zeros(len(fakeTestIndsSelected))])

			predictDX_value[ind], predictDRX_value[ind], decoderR_test_value[ind] = sess.run([predictDX, predictDRX, decoderR_test],
													feed_dict={xTest: test_x})
													
			#--------------
			# 評価値の計算と記録
			recallDX, precisionDX, f1DX = calcEval(predictDX_value[ind][:,0], test_y, threFake)
			recallDRX, precisionDRX, f1DRX = calcEval(predictDRX_value[ind][:,0], test_y, threFake)
		
			recallDXs[ind].append(recallDX)
			precisionDXs[ind].append(precisionDX)
			f1DXs[ind].append(f1DX)
		
			recallDRXs[ind].append(recallDRX)
			precisionDRXs[ind].append(precisionDRX)
			f1DRXs[ind].append(f1DRX)
			#--------------

			#--------------
			print("ratio:%f" % (testFakeRatio))
			print("recallDX=%f, precisionDX=%f, f1DX=%f" % (recallDX, precisionDX, f1DX))
			print("recallDRX=%f, precisionDRX=%f, f1DRX=%f" % (recallDRX, precisionDRX, f1DRX))
			#--------------

			def plotImg(x,y,path):
				#--------------
				# 画像を保存
				plt.close()

				fig, figInds = plt.subplots(nrows=2, ncols=x.shape[0], sharex=True)

				for figInd in np.arange(x.shape[0]):
					fig0 = figInds[0][figInd].imshow(x[figInd,:,:,0])
					fig1 = figInds[1][figInd].imshow(y[figInd,:,:,0])

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
				#--------------


			if ind == 0:

				#--------------
				# 学習で用いている画像（元の画像、ノイズ付加した画像、decoderで復元した画像）を保存
				plt.close()
				fig, figInds = plt.subplots(nrows=3, ncols=nPlotImg, sharex=True)
	
				for figInd in np.arange(figInds.shape[1]):
					fig0 = figInds[0][figInd].imshow(batch_x[figInd,:,:,0])
					fig1 = figInds[1][figInd].imshow(batch_x_fake[figInd,:,:,0])
					fig2 = figInds[2][figInd].imshow(decoderR_train_value[figInd,:,:,0])

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
	
				path = os.path.join(visualPath,"img_train_{}_{}_{}.png".format(postFix,testFakeRatio,ite))
				plt.savefig(path)
				#--------------

				#--------------
				# 提案法で生成した画像（元の画像、提案法で生成たい異常画像）を保存
				if isEmbedSampling:
					path = os.path.join(visualPath,"img_train_aug_{}_{}_{}.png".format(postFix,testFakeRatio,ite))
					plotImg(batch_x[:nPlotImg], decoderR_train_aug_value[:nPlotImg],path)
				#--------------
							
				#--------------
				# 評価画像のうち正常のものを保存
				path = os.path.join(visualPath,"img_test_true_{}_{}_{}.png".format(postFix,testFakeRatio,ite))
				plotImg(test_x[:nPlotImg], decoderR_test_value[ind][:nPlotImg],path)
				#--------------
		
				#--------------
				# 評価画像のうち異常のものを保存
				path = os.path.join(visualPath,"img_test_fake_{}_{}_{}.png".format(postFix,testFakeRatio,ite))
				plotImg(test_x[-nPlotImg:], decoderR_test_value[ind][-nPlotImg:],path)
				#--------------
		
		#--------------
		# チェックポイントの保存
		saver = tf.train.Saver()
		saver.save(sess,"./models/model{}.ckpt".format(postFix))
		#--------------
		
#--------------
# pickleに保存
path = os.path.join(logPath,"log{}.pickle".format(postFix))
with open(path, "wb") as fp:
	pickle.dump(batch_x,fp)
	pickle.dump(batch_x_fake,fp)
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
	pickle.dump(recallDRXs,fp)
	pickle.dump(precisionDRXs,fp)
	pickle.dump(f1DRXs,fp)	
	pickle.dump(lossR_values,fp)
	pickle.dump(lossRAll_values,fp)
	pickle.dump(lossD_values,fp)
	pickle.dump(params,fp)

	if isEmbedSampling:
		pickle.dump(decoderR_train_aug_value,fp)
#--------------
#===========================
