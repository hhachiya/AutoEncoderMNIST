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
# ランダムシード
np.random.seed(0)
#===========================

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
	
# fc layer with ReLU
def fc_relu(inputs, w, b):
	fc = tf.matmul(inputs, w) + b
	fc = tf.nn.relu(fc)
	return fc
	
# fc layer with softmax
def fc_sigmoid(inputs, w, b):
	fc = tf.matmul(inputs, w) + b
	fc = tf.nn.sigmoid(fc)
	return fc
#===========================

#===========================
# エンコーダ
# 画像をz_dim次元のベクトルにエンコード
# reuse=Trueで再利用できる（tf.variable_scope() は，変数の管理に用いるスコープ定義）
def encoderR(x, z_dim, reuse=False):
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
		fc1 = fc_relu(conv2, fcW1, fcB1)
		#--------------
		
		return fc1
#===========================

#===========================
# デコーダ
# z_dim次元の画像にデコード
# reuse=Trueで再利用できる（tf.variable_scope() は，変数の管理に用いるスコープ定義）
def decoderR(z,z_dim,reuse=False):
	with tf.variable_scope('decoderR') as scope:
		if reuse:
			scope.reuse_variables()

		#--------------
		# embeddingベクトルを特徴マップに変換
		# 2次元画像を１次元に変更して全結合層へ渡す
		fcW1 = weight_variable("fcW1", [z_dim, 7*7*32])
		fcB1 = bias_variable("fcB1", [7*7*32])
		fc1 = fc_relu(z, fcW1, fcB1)
		
		batch_size = tf.shape(fc1)[0]
		fc1 = tf.reshape(fc1, tf.stack([batch_size, 7, 7, 32]))
		#--------------
		
		# padding='SAME'のとき、出力のサイズO = 入力サイズI/ストライドS
		# 7 x 2 = 14
		convW1 = weight_variable("convW1", [3, 3, 32, 32])
		convB1 = bias_variable("convB1", [32])
		conv1 = conv2d_t_relu(fc1, convW1, convB1, output_shape=[batch_size,14,14,32], stride=[1,2,2,1])
		
		# 14 x 2 = 28
		convW2 = weight_variable("convW2", [3, 3, 1, 32])
		convB2 = bias_variable("convB2", [1])
		#output = conv2d_t_sigmoid(conv1, convW2, convB2, output_shape=[batch_size,28,28,1], stride=[1,2,2,1])
		output = conv2d_t_relu(conv1, convW2, convB2, output_shape=[batch_size,28,28,1], stride=[1,2,2,1])
		
		return output
#===========================

#===========================
# D Network
# 
# reuse=Trueで再利用できる（tf.variable_scope() は，変数の管理に用いるスコープ定義）
def DNet(x, z_dim=1, reuse=False):
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
		fc1 = fc_sigmoid(conv2, fcW1, fcB1)
		#--------------
		
		return fc1
#===========================

#===========================
# Rのエンコーダとデコーダの連結
xTrue = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
xFake = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
xTest = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])

# 学習用
encoderR_train = encoderR(xFake, z_dim_R)
decoderR_train = decoderR(encoderR_train, z_dim_R)

# テスト用
encoderR_test = encoderR(xTest, z_dim_R, reuse=True)
decoderR_test = decoderR(encoderR_test, z_dim_R, reuse=True)
#===========================

#===========================
# 損失関数の設定

#学習用
predictFake_train = DNet(decoderR_train)
predictTrue_train = DNet(xTrue,reuse=True)
lossR = tf.reduce_mean(tf.square(decoderR_train - xTrue))
lossRAll = tf.reduce_mean(tf.log(1 - predictFake_train + lambdaSmall)) + lambdaR * lossR
lossD = tf.reduce_mean(tf.log(predictTrue_train  + lambdaSmall)) + tf.reduce_mean(tf.log(1 - DNet(decoderR_train,reuse=True) +  lambdaSmall))

# R & Dの変数
Rvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="encoderR") + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="decoderR")
Dvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="DNet")

#--------------
# ランダムシードの設定
tf.set_random_seed(0)
#--------------

'''
trainerR = tf.train.AdamOptimizer(1e-3).minimize(lossR, var_list=Rvars)
trainerRAll = tf.train.AdamOptimizer(1e-3).minimize(lossRAll, var_list=Rvars)
trainerD = tf.train.AdamOptimizer(1e-3).minimize(-lossD, var_list=Dvars)
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
#===========================

#===========================
#テスト用
predictDX = DNet(xTest,reuse=True)
predictDRX = DNet(decoderR_test,reuse=True)
#===========================

#===========================
# メイン
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#--------------
# MNISTのデータの取得
myData = input_data.read_data_sets("MNIST/",dtype=tf.uint8)
#myData = input_data.read_data_sets("MNIST/")
#--------------

#--------------
# テストデータの作成
targetInds = np.where(myData.test.labels == targetChar)[0]

# データの数
fakeNum = int(np.floor(len(targetInds)*testFakeRatio))
targetNum = len(targetInds) - fakeNum

# Fakeのindex（シャッフル）
fakeInds = np.setdiff1d(np.arange(len(myData.test.labels)),targetInds)
fakeInds = fakeInds[np.random.permutation(len(fakeInds))[:fakeNum]]

# Trueのindex（シャッフル）
targetIndsSelected = targetInds[np.random.permutation(len(targetInds))[:targetNum]]

# reshape & concat
test_x = np.reshape(myData.test.images[targetIndsSelected],(len(targetIndsSelected),28,28,1))
test_x_fake = np.reshape(myData.test.images[fakeInds],(len(fakeInds),28,28,1))
test_x = np.vstack([test_x, test_x_fake])
test_y = np.hstack([np.ones(len(targetIndsSelected)),np.zeros(len(fakeInds))])
#--------------

#--------------
# 評価値、損失を格納するリスト
recallDXs = []
precisionDXs = []
f1DXs = []
recallDRXs = []
precisionDRXs = []
f1DRXs = []

lossR_values = []
lossRAll_values = []
lossD_values = []
#--------------

for ite in range(10000):
	
	#--------------
	# 学習データの作成
	batch = myData.train.next_batch(batch_size)
	batch_x_all = np.reshape(batch[0],(batch_size,28,28,1))

	# targetCharのみのデータ
	targetInds = np.where(batch[1] == targetChar)[0]
	batch_x = batch_x_all[targetInds]
	
	# ノイズを追加する(ガウシアンノイズ)
	# 正規分布に従う乱数を出力
	batch_x_fake = batch_x + np.random.normal(0,1,batch_x.shape)
	#--------------

	#--------------
	# 学習
	if trainMode == 0:
		_, lossR_value, lossRAll_value, lossD_value, decoderR_train_value, encoderR_train_value = sess.run(
											[trainerR, lossR, lossRAll, lossD, decoderR_train, encoderR_train],
											feed_dict={xTrue: batch_x,xFake: batch_x_fake})
											
		if lossR_value < threSquaredLoss:
			trainMode = 1

	elif trainMode == 1:
		_, _, lossR_value, lossRAll_value, lossD_value, decoderR_train_value, encoderR_train_value, predictFake_train_value, predictTrue_train_value = sess.run([trainerRAll, trainerD,lossR, lossRAll, lossD, decoderR_train, encoderR_train, predictFake_train, predictTrue_train],feed_dict={xTrue: batch_x,xFake: batch_x_fake})

	# 損失の記録
	lossR_values.append(lossR_value)
	lossRAll_values.append(lossRAll_value)
	lossD_values.append(lossD_value)
	
	if ite%10 == 0:
		print("ite: %d, lossR=%f, lossRAll=%f, lossD=%f" % (ite, lossR_value, lossRAll_value, lossD_value))
	#--------------

	#--------------
	# テスト
	if ite % 200 == 0:
		predictDX_value, predictDRX_value, decoderR_test_value = sess.run([predictDX, predictDRX, decoderR_test], feed_dict={xTest: test_x})

		#--------------
		# 評価値の計算用の関数
		def calcEval(predict, gt, threFake=0.5):
			predict[predict >= threFake] = 1.
			predict[predict < threFake] = 0.
			
			recall = np.sum(predict[gt==1])/np.sum(gt==1)
			precision = np.sum(predict[gt==1])/np.sum(predict==1)
			f1 = 2 * (precision * recall)/(precision + recall)

			return recall, precision, f1
		#--------------
		
		#--------------
		# 評価値の計算と記録
		recallDX, precisionDX, f1DX = calcEval(predictDX_value[:,0], test_y, threFake)
		recallDRX, precisionDRX, f1DRX = calcEval(predictDRX_value[:,0], test_y, threFake)
		
		recallDXs.append(recallDX)
		precisionDXs.append(precisionDX)
		f1DXs.append(f1DX)
		
		recallDRXs.append(recallDRX)
		precisionDRXs.append(precisionDRX)
		f1DRXs.append(f1DRX)
		#--------------

		#--------------
		print("\t recallDX=%f, precisionDX=%f, f1DX=%f" % (recallDX, precisionDX, f1DX))
		print("\t recallDRX=%f, precisionDRX=%f, f1DRX=%f" % (recallDRX, precisionDRX, f1DRX))
		#--------------
		
		#--------------
		# 画像を保存
		plt.close()
		fig, figInds = plt.subplots(nrows=2, ncols=10, sharex=True)
	
		for figInd in np.arange(figInds.shape[1]):
			fig0 = figInds[0][figInd].imshow(test_x[figInd,:,:,0])
			fig1 = figInds[1][figInd].imshow(decoderR_test_value[figInd,:,:,0])

			# ticks, axisを隠す
			fig0.axes.get_xaxis().set_visible(False)
			fig0.axes.get_yaxis().set_visible(False)
			fig0.axes.get_xaxis().set_ticks([])
			fig0.axes.get_yaxis().set_ticks([])
			fig1.axes.get_xaxis().set_visible(False)
			fig1.axes.get_yaxis().set_visible(False)
			fig1.axes.get_xaxis().set_ticks([])
			fig1.axes.get_yaxis().set_ticks([])
	
		path = os.path.join(visualPath,"img{}_{}.png".format(postFix,ite))
		plt.savefig(path)
		#--------------
		
		#--------------
		# 画像を保存
		plt.close()
		fig, figInds = plt.subplots(nrows=2, ncols=10, sharex=True)
	
		for figInd in np.arange(figInds.shape[1]):
			fig0 = figInds[0][figInd].imshow(test_x[-figInd,:,:,0])
			fig1 = figInds[1][figInd].imshow(decoderR_test_value[-figInd,:,:,0])

			# ticks, axisを隠す
			fig0.axes.get_xaxis().set_visible(False)
			fig0.axes.get_yaxis().set_visible(False)
			fig0.axes.get_xaxis().set_ticks([])
			fig0.axes.get_yaxis().set_ticks([])
			fig1.axes.get_xaxis().set_visible(False)
			fig1.axes.get_yaxis().set_visible(False)
			fig1.axes.get_xaxis().set_ticks([])
			fig1.axes.get_yaxis().set_ticks([])
	
		path = os.path.join(visualPath,"img_fake_{}_{}.png".format(postFix,ite))
		plt.savefig(path)
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
#--------------
#===========================
