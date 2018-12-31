import util
import numpy as np
import os
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

class ChessModel:
	def __init__(self, res_layer_num = 5, epochs = 1):
		self.model = None
		self.res_layer_num = res_layer_num
		self.epochs = epochs
		data = np.load(os.path.join("processed", "dataset_keras_339.npz"))
		self.state_array = data['arr_0']
		self.state_array = np.moveaxis(self.state_array, 1, 3)
		self.policy_array = data['arr_1']
		self.value_array = data['arr_2']
		print("[INFO] Training data: State: {}, Policy: {}, Value: {}".format(self.state_array.shape, self.policy_array.shape, self.value_array.shape), '\n\n')

	def build(self):
		input_shape = self.state_array.shape[1:]
		input_x = x = Input(shape = input_shape)

		x = Conv2D(filters = 256, kernel_size = 5, padding = "same", 
			       kernel_regularizer = l2(1e-4), name = "input_conv_5x5_256")(x)
		x = BatchNormalization(axis = -1, name = "input_batchnorm")(x)
		x = Activation("relu", name="input_relu")(x)

		for i in range(self.res_layer_num):
			x = self.build_residual_block(x, i + 1)

		res_out = x

		# for policy output
		x = Conv2D(filters = 2, kernel_size = 1, kernel_regularizer = l2(1e-4), name = "policy_conv_1x1_2")(res_out)
		x = BatchNormalization(axis = -1, name = "policy_batchnorm")(x)
		x = Activation("relu", name = "policy_relu")(x)
		x = Flatten(name = "policy_flatten")(x)
		policy_out = Dense(units = util.labels_n, activation="softmax", kernel_regularizer = l2(1e-4), name="policy_out")(x)

		# for value output
		x = Conv2D(filters = 4, kernel_size = 1, kernel_regularizer = l2(1e-4), name = "policy_conv_1x1_4")(res_out)
		x = BatchNormalization(axis = -1, name = "value_batchnorm")(x)
		x = Activation("relu", name = "value_relu")(x)
		x = Flatten(name = "value_flatten")(x)
		x = Dense(units = 256, kernel_regularizer = l2(1e-4), activation = "relu", name = "value_dense")(x)
		value_out = Dense(units = 1, kernel_regularizer = l2(1e-4), activation = "tanh", name = "value_out")(x)

		self.model = Model(inputs = input_x, outputs = [policy_out, value_out], name = "Chess Model")

		plot_model(self.model, to_file='model.png', show_shapes = True)

	def build_residual_block(self, x, index):
		in_x = x
		res_name = "res"+str(index)

		x = Conv2D(filters = 256, kernel_size = 3, padding = "same", 
			       kernel_regularizer = l2(1e-4), name = res_name + "conv1_3x3_256")(x)
		x = BatchNormalization(axis = -1, name = res_name + "_batchnorm1")(x)
		x = Activation("relu",name = res_name + "_relu1")(x)

		x = Conv2D(filters = 256, kernel_size = 3, padding = "same",
			       kernel_regularizer = l2(1e-4), name = res_name + "conv2_3x3_256")(x)
		x = BatchNormalization(axis = -1, name = res_name + "_batchnorm2")(x)
		x = Add(name = res_name + "_add")([in_x, x])
		x = Activation("relu",name = res_name + "_relu2")(x)
		return x

	def compile_model(self):
		if os.path.isfile(os.path.join("nets", "keras_model_weights_{}_{}e.h5".format(len(self.state_array), self.epochs))):
			print("[INFO] Loading model for the saved file")
			self.model.load_weights(os.path.join("nets", "keras_model_weights_{}_{}e.h5".format(len(self.state_array), self.epochs)))
		else:
			print("[INFO] Starting the model from scratch")

		loss_weights = [1.25, 1.0] # [policy, value] prevent value overfit in SL
		opt = Adam()
		losses = ['categorical_crossentropy', 'mean_squared_error']
		self.model.compile(optimizer = opt, loss = losses, loss_weights = loss_weights)

	def training(self, batch_size = 256):
		self.compile_model()
		tensorboard_callback = TensorBoard(log_dir="./logs", batch_size = batch_size, histogram_freq = 1)
		self.model.fit(x = self.state_array, y = [self.policy_array, self.value_array], batch_size = batch_size,
			           epochs = self.epochs, shuffle = True, validation_split=0.02, callbacks=[tensorboard_callback])
		self.model.save_weights(os.path.join("nets", "keras_model_weights_{}_{}e.h5".format(len(self.state_array), epochs)))


model = ChessModel(epochs = 100)
model.build()
model.training()