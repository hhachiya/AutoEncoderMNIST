# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.ops import nn_ops
import numpy as np
import math, os
import pickle
import pdb
import input_data
import matplotlib.pylab as plt

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
def conv2d_t_relu(inputs, w, b, output_shape, stride):
	conv = tf.nn.conv2d_transpose(inputs, w, output_shape=output_shape, strides=stride, padding='SAME') + b
	conv = tf.nn.relu(conv)
	return conv

# fc layer
def fc_relu(inputs, w, b):
	fc = tf.matmul(inputs, w) + b
	fc = tf.nn.relu(fc)
	return fc
#===========================

#===========================
# エンコーダ
# 画像をz_dim次元のベクトルにエンコード
# reuse=Trueで再利用できる（tf.variable_scope() は，変数の管理に用いるスコープ定義）
def encoderImg(x, z_dim, reuse=False):
	with tf.variable_scope('encoderImg') as scope:
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
def decoderImg(z,z_dim,reuse=False):
	with tf.variable_scope('decoderImg') as scope:
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
		convB2 = bias_variable("convB2", [32])
		output = conv2d_t_relu(conv1, convW2, convB2, output_shape=[batch_size,28,28,1], stride=[1,2,2,1])
		
		return output
		
#===========================
# kernelの計算
def compute_kernel(x, y):
	x_size = tf.shape(x)[0]
	y_size = tf.shape(y)[0]
	dim = tf.shape(x)[1]
	tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
	tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
	return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))
#===========================

#===========================
# mmdの計算
def compute_mmd(x, y):
	x_kernel = compute_kernel(x, x)
	y_kernel = compute_kernel(y, y)
	xy_kernel = compute_kernel(x, y)
	return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)
#===========================

#===========================
# エンコーダとデコーダの連結
z_img_dim = 100
x_img = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])

# 学習用
train_z_img_op = encoderImg(x_img, z_img_dim)
train_xr_img_op = decoderImg(train_z_img_op, z_img_dim)

# テスト用
test_z_img_op = encoderImg(x_img, z_img_dim, reuse=True)
test_xr_img_op = decoderImg(test_z_img_op, z_img_dim, reuse=True)
#===========================

#===========================
# maximum mean discrepancyの計算
mmd_sample_num = 200
true_img_samples = tf.random_normal(tf.stack([mmd_sample_num, z_img_dim]))
loss_mmd_img = compute_mmd(true_img_samples, train_z_img_op)
#===========================

#===========================
# 損失関数の設定
loss_nll_img = tf.reduce_mean(tf.square(train_xr_img_op - x_img))

# 二乗損失とMMD損失の和
loss_img = loss_nll_img + loss_mmd_img
#loss_img = loss_nll_img
trainer_img = tf.train.AdamOptimizer(1e-3).minimize(loss_img)
#===========================

#===========================
# メイン
visualPath = 'visualization'
modelPath = 'models'

# MNISTのデータの取得
myImage = input_data.read_data_sets("MNIST/",one_hot=True)
batch_size = 200

# TF初期化
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(3000):
	
	# 学習データの作成
	batch = myImage.train.next_batch(batch_size)
	batch_x_img = np.reshape(batch[0],(batch_size,28,28,1))

	# パラメータ更新
	_, nll_img, train_xr_img, train_z_img = sess.run([trainer_img, loss_nll_img, train_xr_img_op, train_z_img_op], feed_dict={x_img: batch_x_img})
	print("Image, iteration: %d, Negative log likelihood is %f" % (i,nll_img))

	if i % 50 == 0:

		# テストデータの作成
		batch_test = myImage.test.next_batch(batch_size)
		batch_x_test_img = np.reshape(batch_test[0],(batch_size,28,28,1))

		# テストの実行
		test_xr_img,test_z_img = sess.run([test_xr_img_op,test_z_img_op], feed_dict={x_img: batch_x_test_img})

		#--------------
		# pickleに保存
		path = os.path.join(visualPath,"img_{}.pickle".format(i))
		with open(path, "wb") as fp:
			pickle.dump(batch_x_img,fp)  
			pickle.dump(train_xr_img,fp)
			pickle.dump(train_z_img,fp)
			pickle.dump(batch_x_test_img,fp)
			pickle.dump(test_xr_img,fp)
			pickle.dump(test_z_img,fp)
		#--------------
			
		#--------------
		# 画像を保存
		fig, figInds = plt.subplots(nrows=2, ncols=10, sharex=True)
	
		for figInd in np.arange(figInds.shape[1]):
			fig0 = figInds[0][figInd].imshow(batch_x_test_img[figInd,:,:,0])
			fig1 = figInds[1][figInd].imshow(test_xr_img[figInd,:,:,0])

			# ticks, axisを隠す
			fig0.axes.get_xaxis().set_visible(False)
			fig0.axes.get_yaxis().set_visible(False)
			fig0.axes.get_xaxis().set_ticks([])
			fig0.axes.get_yaxis().set_ticks([])
			fig1.axes.get_xaxis().set_visible(False)
			fig1.axes.get_yaxis().set_visible(False)
			fig1.axes.get_xaxis().set_ticks([])
			fig1.axes.get_yaxis().set_ticks([])
	
		path = os.path.join(visualPath,"img_sample_{}.png".format(i))
		

				
		plt.savefig(path)
		#--------------

		#--------------
		# チェックポイントの保存
		saver = tf.train.Saver()
		saver.save(sess,"./models/img_{}.ckpt".format(i))
		#--------------

#---------------------

