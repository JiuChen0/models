import numpy as np
import tensorflow as tf
from utils import FullConnect, MultiHeadsAtten, risk

# from tensorflow.keras.layers import LayerNormalization, Dense, Dropout, MultiHeadAttention

'''
num_layers is the parameter for specifying how many iterations the encoder block should 
have. d_model is the dimensionality of the input, num_heads is the number of attention heads, 
and dff is the dimensionality of the feed-forward network. The rate parameter is for the dropout rate.
'''

'''Encoder Block 
input -> SelfAttention + Add&Norm (mha) -> FeedForward + Add&Norm (fc) '''
class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(TransformerEncoderBlock, self).__init__()
        self.mha = MultiHeadsAtten()
        self.fc = FullConnect()
    def forward(self, x):
        atten_score = self.mha(x, x, x, Type='E')   # Type='E' -> Encoder_Attention
        out = self.fc(atten_score) 
        return out

'''Decoder Block 
input -> SelfAttention + Add&Norm (mha) -> Encoder-DecoderAttention + Add&Norm (mha) -> FeedForward + Add&Norm (fc) '''
class TransformerDecoderBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(TransformerDecoderBlock, self).__init__()
        self.mha = MultiHeadsAtten()
        self.fc = FullConnect()
    def call(self, y, enc_output):  
        attn1_score = self.mha(y, y, y, Type='D')  # Self attention (Look_Ahead_Mask=True)
        attn2_score = self.mha(attn1_score, enc_output, enc_output, Type='D-E')  # Encoder-decoder attention (Look_Ahead_Mask=False)
        out = self.fc(attn2_score) 
        return out

'''Set the Number of Encoder & Decoder Layer (default: num_layer = 3)'''
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.enc_layers = [TransformerEncoderBlock() 
                           for _ in range(num_layers)]
    def call(self, x):
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        return x  # (batch_size, input_seq_len, d_model)

class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, num_layers):
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.dec_layers = [TransformerDecoderBlock() 
                           for _ in range(num_layers)]
    def call(self, y, enc_output):
        for i in range(self.num_layers):
            y = self.dec_layers[i](y, enc_output)
        return y

'''The Main Model'''
class MyModel_TransLTEE(tf.keras.Model):
    def __init__(self, t0, t, input_dim=100, num_layers=3):
        super(MyModel_TransLTEE, self).__init__()
        # read treatment and recognize the position
        self.t0 = t0
        self.t = t
        self.i0 = tf.cast(tf.where(t < 1)[:,0], tf.int32)
        self.i1 = tf.cast(tf.where(t > 0)[:,0], tf.int32)
        self.regularizer = tf.keras.regularizers.l2(l2=1.0)
        self.input_phi = tf.keras.layers.Dense(input_dim, activation='relu', kernel_regularizer=self.regularizer)
        self.transformer_encoder = TransformerEncoder(num_layers=num_layers)
        self.transformer_decoder = TransformerDecoder(num_layers=num_layers)
        self.dense = tf.keras.layers.Dense(100)
        self.linear = tf.keras.layers.Dense(1)
        # self.softmax = tf.keras.layers.Softmax()

    def call(self, x, tar_input, tar_real):
        # Divide x into control(w=0) group and treated(w=1) group
        x_0 = tf.gather(x[:,:], self.i0)
        x_1 = tf.gather(x[:,:], self.i1)
        tar_0 = self.dense(tf.expand_dims(tf.gather(tar_input[:,:], self.i0),-1))
        tar_1 = self.dense(tf.expand_dims(tf.gather(tar_input[:,:], self.i1),-1))
        tar_real_0 = tf.gather(tar_real[:,:], self.i0)
        tar_real_1 = tf.gather(tar_real[:,:], self.i1)
        phi_0 = self.input_phi(x_0)
        phi_1 = self.input_phi(x_1)

        '''Control Group (w=0)'''
        encoded0 = self.transformer_encoder(phi_0)
        encoded0 = tf.repeat(encoded0[:, np.newaxis, :], self.t0, axis=1)    # [n, 100] -> [n, time, 100]
        print(tar_0.shape, encoded0.shape)
        decoded0 = self.transformer_decoder(tar_0, encoded0)
        output_0 = self.linear(decoded0)

        '''Treated Group (w=1)'''
        encoded1 = self.transformer_encoder(phi_1)
        encoded1 = tf.repeat(encoded1[:, np.newaxis, :], self.t0, axis=1)    # [n, 100] -> [n, time, 100]
        decoded1 = self.transformer_decoder(tar_1, encoded1)
        output_1 = self.linear(decoded1)

        '''Concat two groups'''
        encoded = tf.concat((encoded0, encoded1), axis=0)
        output = tf.concat((output_0, output_1), axis=0)
        tar_real = tf.concat((tar_real_0,tar_real_1), axis=0)

        '''Evaluate and Return the Predicted Error & Wasserstain Distance'''
        predicted_error = risk().pred_error(output, tar_real)
        dis = risk().distance(encoded, self.t0, self.t)
        # # output = self.softmax(linear_output)
        return output, predicted_error, dis


