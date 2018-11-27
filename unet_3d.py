from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose, Dropout, ReLU, Cropping3D, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras import regularizers 
from keras.layers.normalization import BatchNormalization as bn

def conv_3d(input_tensor, n_filters, kernel_size = (3, 3, 3), strides = (1, 1, 1), activation = "relu", padding = "valid", batch_norm = True, dropout = True):
  
  conv = Conv3D(n_filters, kernel_size, padding = padding, strides = strides)(input_tensor)
  
  if batch_norm:
    conv = bn()(conv)
    
  if activation.lower() == "relu":
    conv = ReLU()(conv)
  
  if dropout:
    conv = Dropout(0.3)(conv)
    
  return conv

def upconv_and_concat(tensor_to_upconv, tensor_to_concat, upconv_n_filters, kernel_size = (2, 2, 2), strides = (2, 2, 2), padding = "valid"):
  upconv = Conv3DTranspose(upconv_n_filters, kernel_size, strides = strides, padding = padding)(tensor_to_upconv)
  
  crop_size = (int(tensor_to_concat.shape[1]) - int(tensor_to_upconv.shape[1])*2) // 2
  cropped = Cropping3D((crop_size, crop_size, crop_size))(tensor_to_concat)
  concat = concatenate([upconv, cropped], axis = 4)
  
  return concat

def unet_3d(input_shape, n_classes, loss, metrics, n_gpus = 1, optimizer = "adam", lr = 0.0001, batch_norm = True, activation = "relu", pool_size = (2, 2, 2)):
  
  # Encoder
  input = Input(input_shape)
  conv1_1 = conv_3d(input, 32, batch_norm = batch_norm, activation = activation)
  conv1_2 = conv_3d(conv1_1, 32, batch_norm = batch_norm, activation = activation)
  pool_1 = MaxPooling3D(pool_size)(conv1_2)
  

  conv2_1 = conv_3d(pool_1, 64, batch_norm = batch_norm, activation = activation)
  conv2_2 = conv_3d(conv2_1, 64, batch_norm = batch_norm, activation = activation)
  pool_2 = MaxPooling3D(pool_size)(conv2_2)
  
  conv3_1 = conv_3d(pool_2, 128, batch_norm = batch_norm, activation = activation)
  conv3_2 = conv_3d(conv3_1, 128, batch_norm = batch_norm, activation = activation)
  pool_3 = MaxPooling3D(pool_size)(conv3_2)
  
  conv4_1 = conv_3d(pool_3, 256, batch_norm = batch_norm, activation = activation)
  conv4_2 = conv_3d(conv4_1, 128, batch_norm = batch_norm, activation = activation)
  
  
  # Decoder
  upconv_5 = upconv_and_concat(conv4_2, conv3_2, 128)
  conv5_1 = conv_3d(upconv_5, 128, batch_norm = batch_norm, activation = activation)
  conv5_2 = conv_3d(conv5_1, 64, batch_norm = batch_norm, activation = activation)
  
  upconv_6 = upconv_and_concat(conv5_2, conv2_2, 64)
  conv6_1 = conv_3d(upconv_6, 64, batch_norm = batch_norm, activation = activation)
  conv6_2 = conv_3d(conv6_1, 32, batch_norm = batch_norm, activation = activation)
  
  upconv_7 = upconv_and_concat(conv6_2, conv1_2, 32)
  conv7_1 = conv_3d(upconv_7, 32, batch_norm = batch_norm, activation = activation)
  conv7_2 = conv_3d(conv7_1, 32, batch_norm = batch_norm, activation = activation)
  
  final_conv = Conv3D(n_classes, kernel_size = (1, 1, 1), padding = "same")(conv7_2)
  
  
  model = Model(input, final_conv)
  
  if optimizer == "adam":
    adam = Adam(lr = lr)
    
    model.compile(optimizer = adam, loss = loss, metrics = metrics)
  
  else:
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
  
  return model
