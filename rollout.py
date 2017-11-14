"""
conditionalGan
encoder-decoder
encoderは普通の
deocoderは自分でパラメータを作成??
"""

import copy
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.ops import rnn
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import tensor_array_ops, control_flow_ops


class LSTM:
    def __init__(self, lstm, update_rate, given_num):
        self.lstm =lstm
        self.update_rate = update_rate
        self.given_num = given_num

        self.num_emb = self.lstm.num_emb
        self.batch_size = self.lstm.batch_size
        self.emb_dim = self.lstm.emb_dim # word_id -> [emb_dim]
        self.hidden_dim = self.lstm.hidden_dim # [emb_dim] -> [hidden_dim]
        self.sequence_length = self.lstm.sequence_length
        self.start_token = self.lstm.start_token

        self.embeddings = tf.identity(self.lstm.embeddings)
        self.recurrent_unit = self.create_recurrent_unit()
        self.output_unit = self.create_output_unit()


        self.init_gen_o = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                        dynamic_size=False, infer_shape=True)
        self.init_gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)
        self.init_predictions = tensor_array_ops.TensorArray(
                                            dtype=tf.float32, size=self.sequence_length,
                                            dynamic_size=False, infer_shape=True)
        self.ta_emb_x = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length)
        self.ta_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length)

    def __call__(self, c_0, h_0, inputs, generate_flag=False):
        self.generate_flag = generate_flag
        self.inputs = inputs
        self.processed_x = self.return_processed_x(inputs) # seq_length x batch_size x emb_dim

        h_0 = tf.stack([h_0, c_0])
        x_0 = tf.nn.embedding_lookup(self.embeddings, self.start_token)

        gen_x = self.init_gen_x

        self.ta_emb_x = self.ta_emb_x.unstack(self.processed_x)
        self.ta_x = self.ta_x.unstack(inputs)
        #self.ta_x = self.ta_x.unstack(tf.transpose(inputs, perm=[1,0]))

        i, x_t, h_t, given_num, self.gen_x = control_flow_ops.while_loop(
            cond = lambda i, _1, _2, given_num, _4: i < given_num,
            body = self.recurrence1,
            loop_vars = (tf.constant(0, dtype=tf.int32),
                        x_0, h_0, self.given_num, gen_x))

        _, _, h_t, _, self.gen_x = control_flow_ops.while_loop(
            cond = lambda i, _1, _2, _3, _4: i < self.sequence_length,
            body = self.recurrence2,
            loop_vars = (i, x_t, h_t, given_num, self.gen_x)
        )

        self.gen_x = tf.transpose(self.gen_x.stack(), perm=[1,0]) # (batch_size, seq_length)

        # return self.gen_x, self.predictions, self.gen_o, self.h_t
        return self.gen_x, h_t

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)

    def return_processed_x(self, x):
        processed_x = tf.nn.embedding_lookup(self.embeddings, x)  # seq_length x batch_size x emb_dim
        return processed_x

    def create_recurrent_unit(self):
        self.Wi = tf.identity(self.lstm.Wi)
        self.Ui = tf.identity(self.lstm.Ui)
        self.bi = tf.identity(self.lstm.bi)

        self.Wf = tf.identity(self.lstm.Wf)
        self.Uf = tf.identity(self.lstm.Uf)
        self.bf = tf.identity(self.lstm.bf)

        self.Wog = tf.identity(self.lstm.Wog)
        self.Uog = tf.identity(self.lstm.Uog)
        self.bog = tf.identity(self.lstm.bog)

        self.Wc = tf.identity(self.lstm.Wc)
        self.Uc = tf.identity(self.lstm.Uc)
        self.bc = tf.identity(self.lstm.bc)

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)
            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )
            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
            )
            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )
            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )
            # Final Memory cell
            c = f * c_prev + i * c_
            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit

    def create_output_unit(self):
        self.Wo = tf.identity(self.lstm.Wo)
        self.bo = tf.identity(self.lstm.bo)

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit


    # iがgiven_numより小さい時は提供されているtokenをinputとする
    def recurrence1(self, i, x_t, h_prev, given_num, gen_x):
        h_t = self.recurrent_unit(x_t, h_prev)
        #o_t = self.output_unit(h_t)
        #prob = tf.nn.softmax(o_t)
        #next_token = tf.cast(tf.reshape(tf.multinomial(tf.log(prob), 1), [self.batch_size]), tf.int32) # token_id
        x_next = self.ta_emb_x.read(i)
        gen_x = gen_x.write(i, self.ta_x.read(i))
        #gen_x = gen_x.write(i, next_token)
        x_next.set_shape([self.batch_size, self.emb_dim])
        return i+1, x_next, h_t, given_num, gen_x

    # iがgiven_num以上の場合はroll-out開始, 時刻tのoutputを時刻t+1のinputとする
    def recurrence2(self, i, x_t, h_prev, given_num, gen_x):
        h_t = self.recurrent_unit(x_t, h_prev)
        o_t = self.output_unit(h_t)
        prob = tf.nn.softmax(o_t)
        next_token = tf.cast(tf.reshape(tf.multinomial(tf.log(prob), 1), [self.batch_size]), tf.int32)
        x_next = tf.nn.embedding_lookup(self.embeddings, next_token)
        gen_x = gen_x.write(i, next_token)
        h_t.set_shape([2, self.batch_size, self.hidden_dim])
        x_next.set_shape([self.batch_size, self.emb_dim])
        return i+1, x_next, h_t, given_num, gen_x


    def update_recurrent_unit(self) :
        self.Wi = self.update_rate * self.Wi + (1-self.update_rate)*tf.identity(self.lstm.Wi)
        self.Ui = self.update_rate * self.Ui + (1-self.update_rate)*tf.identity(self.lstm.Ui)
        self.bi = self.update_rate * self.bi + (1-self.update_rate)*tf.identity(self.lstm.bi)

        self.Wf = self.update_rate * self.Wf + (1-self.update_rate)*tf.identity(self.lstm.Wf)
        self.Uf = self.update_rate * self.Uf + (1-self.update_rate)*tf.identity(self.lstm.Uf)
        self.bf = self.update_rate * self.bf + (1-self.update_rate)*tf.identity(self.lstm.bf)

        self.Wog = self.update_rate * self.Wog + (1-self.update_rate)*tf.identity(self.lstm.Wog)
        self.Uog = self.update_rate * self.Uog + (1-self.update_rate)*tf.identity(self.lstm.Uog)
        self.bog = self.update_rate * self.bog + (1-self.update_rate)*tf.identity(self.lstm.bog)

        self.Wc = self.update_rate * self.Wc + (1-self.update_rate)*tf.identity(self.lstm.Wc)
        self.Uc = self.update_rate * self.Uc + (1-self.update_rate)*tf.identity(self.lstm.Uc)
        self.bc = self.update_rate * self.bc + (1-self.update_rate)*tf.identity(self.lstm.bc)

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)
            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )
            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
            )
            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )
            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )
            # Final Memory cell
            c = f * c_prev + i * c_
            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit


    def update_output_unit(self):
        self.Wo = self.update_rate * self.Wo + (1-self.update_rate) * tf.identity(self.lstm.Wo)
        self.bo = self.update_rate * self.bo + (1-self.update_rate) * tf.identity(self.lstm.bo)

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            return logits
        return unit


    def update_params(self):
        self.embeddings = tf.identity(self.lstm.embeddings)
        self.recurrent_unit = self.update_recurrent_unit()
        self.output_unit = self.update_output_unit()


class Encoder:
    def __init__(self, lstm, update_rate, given_num):
        self.gen_lstm = lstm # generator_encoderのlstm
        self.ro_lstm = LSTM(lstm, update_rate, given_num)


    def __call__(self, encoder_inputs, generate_flag):

        h0 = tf.zeros([self.gen_lstm.batch_size, self.gen_lstm.hidden_dim])
        _, h_t = self.ro_lstm(h0, h0, encoder_inputs, generate_flag)
        h_t, c_t = tf.unstack(h_t)
        return h_t, c_t



class Decoder:
    def __init__(self, lstm, update_rate, given_num):
        self.gen_lstm = lstm # generator_decoderのlstm
        self.ro_lstm = LSTM(lstm, update_rate, given_num)


    def __call__(self, c, h, decoder_inputs, generate_flag):
        generateds, h_t = self.ro_lstm(c, h, decoder_inputs, generate_flag)
        return generateds, h_t


class Rollout:
    def __init__(self, generator, update_rate):
        self.update_rate = update_rate
        self.generator = generator
        self.generator_encoder = generator.encoder
        self.generator_decoder = generator.decoder


    def return_placeholders(self):
        encoder_inputs = list()
        decoder_inputs = list()
        labels = list()
        weights = list()
        #print(self.enc_seq_length)
        for _ in range(self.generator.enc_seq_length):
            encoder_inputs.append(tf.placeholder(tf.int32, shape=(None, )))
        for _ in range(self.generator.dec_seq_length):
            decoder_inputs.append(tf.placeholder(tf.int32, shape=(None,)))
            labels.append(tf.placeholder(tf.int32, shape=(None,)))
            weights.append(tf.placeholder(tf.float32, shape=(None, )))

        given_num = tf.placeholder(tf.int32)
        return encoder_inputs, decoder_inputs, labels, weights, given_num


    def output(self, encoder_inputs, decoder_inputs, z, given_num, generate_flag):
        # encode
        self.encoder = Encoder(self.generator_encoder.lstm, self.update_rate, given_num)
        h, c = self.encoder(encoder_inputs, generate_flag)

        h = tf.concat([h, z], 1)
        c = tf.concat([c, z], 1)
        # decode
        self.decoder = Decoder(self.generator_decoder.lstm, self.update_rate, given_num)
        generateds, _ = self.decoder(c, h, decoder_inputs, generate_flag)
        return generateds


    def update(self):
        self.encoder.ro_lstm.update_params()
        self.decoder.ro_lstm.update_params()



# end
