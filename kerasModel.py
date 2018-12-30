import util
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.utils import plot_model

class ChessModel:
	def __init__(self, res_layer_num = 5):
		self.model = None
		self.res_layer_num = res_layer_num

	def build(self):
		input_x = x = Input(shape = (18, 8, 8))

		x = Conv2D(filters = 256, kernel_size = 5, padding = "same", data_format = "channels_first", 
			       kernel_regularizer = l2(1e-4), name = "input_conv_5x5_256")(x)
		x = BatchNormalization(axis = 1, name = "input_batchnorm")(x)
		x = Activation("relu", name="input_relu")(x)

		for i in range(self.res_layer_num):
			x = self.build_residual_block(x, i + 1)

		res_out = x

		# for policy output
		x = Conv2D(filters = 2, kernel_size = 1, data_format = "channels_first", kernel_regularizer = l2(1e-4), name = "policy_conv_1x1_2")(res_out)
		x = BatchNormalization(axis = 1, name = "policy_batchnorm")(x)
		x = Activation("relu", name = "policy_relu")(x)
		x = Flatten(name = "policy_flatten")(x)
		policy_out = Dense(units = util.labels_n, activation="softmax", kernel_regularizer = l2(1e-4), name="policy_out")(x)

		# for value output
		x = Conv2D(filters = 4, kernel_size = 1, data_format = "channels_first", kernel_regularizer = l2(1e-4), name = "policy_conv_1x1_4")(res_out)
		x = BatchNormalization(axis = 1, name = "value_batchnorm")(x)
		x = Activation("relu", name = "value_relu")(x)
		x = Flatten(name = "value_flatten")(x)
		x = Dense(units = 256, kernel_regularizer = l2(1e-4), activation = "relu", name = "value_dense")(x)
		value_out = Dense(units = 1, kernel_regularizer = l2(1e-4), activation = "tanh", name = "value_out")(x)

		self.model = Model(inputs = input_x, outputs = [policy_out, value_out], name = "Chess Model")

		plot_model(self.model, to_file='model.png', show_shapes = True)

	def build_residual_block(self, x, index):
		in_x = x
		res_name = "res"+str(index)

		x = Conv2D(filters = 256, kernel_size = 3, padding = "same", data_format = "channels_first", 
			       kernel_regularizer = l2(1e-4), name = res_name + "conv1_3x3_256")(x)
		x = BatchNormalization(axis = 1, name = res_name + "_batchnorm1")(x)
		x = Activation("relu",name = res_name + "_relu1")(x)

		x = Conv2D(filters = 256, kernel_size = 3, padding = "same", data_format = "channels_first", 
			       kernel_regularizer = l2(1e-4), name = res_name + "conv2_3x3_256")(x)
		x = BatchNormalization(axis = 1, name = res_name + "_batchnorm2")(x)
		x = Add(name = res_name + "_add")([in_x, x])
		x = Activation("relu",name = res_name + "_relu2")(x)
		return x

model = ChessModel().build()