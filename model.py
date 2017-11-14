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
    def __init__(self, vocab_size, batch_size, embedding_dim, hidden_dim, seq_length, start_token, embeddings=None):
        self.num_emb = vocab_size
        self.batch_size = batch_size
        self.emb_dim = embedding_dim # word_id -> [emb_dim]
        self.hidden_dim = hidden_dim # [emb_dim] -> [hidden_dim]
        self.sequence_length = seq_length
        self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
        self.embeddings = embeddings

        self.params = []

        if self.embeddings == None:
            self.embeddings = tf.Variable(self.init_matrix([self.num_emb, self.emb_dim]))
            self.params.append(self.embeddings)

        self.recurrent_unit = self.create_recurrent_unit(self.params)
        self.output_unit = self.create_output_unit(self.params)


        self.init_gen_o = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                        dynamic_size=False, infer_shape=True)
        self.init_gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)
        self.init_predictions = tensor_array_ops.TensorArray(
                                            dtype=tf.float32, size=self.sequence_length,
                                            dynamic_size=False, infer_shape=True)
        self.ta_emb_x = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length)

    def __call__(self, c_0, h_0, inputs, generate_flag=False):
        self.generate_flag = generate_flag
        self.inputs = inputs
        self.processed_x = self.return_processed_x(inputs) # seq_length x batch_size x emb_dim

        h_0 = tf.stack([h_0, c_0])
        x_0 = tf.nn.embedding_lookup(self.embeddings, self.start_token)
        gen_o = self.init_gen_o
        gen_x = self.init_gen_x
        predictions = self.init_predictions

        self.ta_emb_x = self.ta_emb_x.unstack(self.processed_x)

        # batch_size分gen_xを作って返すようにすれば良い?
        _, _, self.h_t, self.gen_o, self.gen_x, self.predictions = control_flow_ops.while_loop(
            cond = lambda i, _1, _2, _3, _4, _5: i < self.sequence_length,
            body = self.recurrence,
            loop_vars = (tf.constant(0, dtype=tf.int32),
                        x_0, h_0, gen_o, gen_x, predictions)
        )

        self.gen_x = tf.transpose(self.gen_x.stack(), perm=[1,0]) # (batch_size, seq_length)
        self.gen_o = tf.transpose(self.gen_o.stack(), perm=[1,0])
        #self.predictions = tf.transpose(self.predictions.stack(), perm=[1,0,2]) #(batch_size x seq_length x vocab_size)
        self.predictions = self.predictions.stack()
        return self.gen_x, self.predictions, self.gen_o, self.h_t

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)

    def return_processed_x(self, x):
        processed_x = tf.nn.embedding_lookup(self.embeddings, x)  # seq_length x batch_size x emb_dim
        return processed_x

    def create_recurrent_unit(self, params):
        # Weights and Bias for input and hidden tensor
        self.Wi = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Ui = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bi = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wf = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uf = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bf = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wog = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uog = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bog = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wc = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uc = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bc = tf.Variable(self.init_matrix([self.hidden_dim]))
        params.extend([self.Wi, self.Ui, self.bi,
            self.Wf, self.Uf, self.bf,
            self.Wog, self.Uog, self.bog,
            self.Wc, self.Uc, self.bc])

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

    def create_output_unit(self, params):
        self.Wo = tf.Variable(self.init_matrix([self.hidden_dim, self.num_emb]))
        self.bo = tf.Variable(self.init_matrix([self.num_emb]))
        params.extend([self.Wo, self.bo])

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit


    def recurrence(self, i, x_t, h_prev, gen_o, gen_x, predictions):
        h_t = self.recurrent_unit(x_t, h_prev)
        o_t = self.output_unit(h_t)
        prob = tf.nn.softmax(o_t)
        next_token = tf.cast(tf.reshape(tf.multinomial(tf.log(prob), 1), [self.batch_size]), tf.int32) # token_id
        if self.generate_flag: # generate or test
            x_next = tf.nn.embedding_lookup(self.embeddings, next_token) # (batch_size, emb_dim)
        else: # train
            x_next = self.ta_emb_x.read(i)# (batch_size, emb_dim)
        o = tf.reduce_sum(tf.multiply(tf.one_hot(next_token, self.num_emb, 1.0, 0.0), prob), 1)

        gen_o = gen_o.write(i, o) # [batch_size]
        gen_x = gen_x.write(i, next_token)
        predictions = predictions.write(i, prob)
        x_next.set_shape([self.batch_size, self.emb_dim])
        return i+1, x_next, h_t, gen_o, gen_x, predictions


class Encoder:
    def __init__(self, vocab_size, batch_size, embedding_dim, hidden_dim, enc_seq_length, start_token):
        self.batch_size=batch_size
        self.hidden_dim=hidden_dim
        self.enc_seq_length=enc_seq_length
        self.lstm = LSTM(vocab_size, batch_size, embedding_dim, hidden_dim, enc_seq_length, start_token)

    def __call__(self, encoder_inputs, generate_flag):
        h0 = tf.zeros([self.batch_size, self.hidden_dim])
        _, _, _, h_t = self.lstm(h0, h0, encoder_inputs, generate_flag)
        h_t, c_t = tf.unstack(h_t)
        return h_t, c_t


class Decoder:
    def __init__(self, vocab_size, batch_size, embedding_dim, hidden_dim, dec_seq_length, start_token, embeddings=None):
        self.dec_seq_length = dec_seq_length
        self.lstm = LSTM(vocab_size, batch_size, embedding_dim, hidden_dim, dec_seq_length, start_token, embeddings)


    def __call__(self, c, h, decoder_inputs, generate_flag):
        # decodeするための初期化
        generateds, predictions, outputs, h_t = self.lstm(c, h, decoder_inputs, generate_flag)
        return generateds, predictions, outputs, h_t


class Seq2Seq:
    def __init__(self, vocab_size, batch_size, embedding_dim, hidden_size,
        enc_seq_length, dec_seq_length, zdim, learning_rate=0.01, start_token=0):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.enc_seq_length = enc_seq_length
        self.dec_seq_length = dec_seq_length
        self.start_token = start_token
        self.learning_rate = learning_rate
        self.zdim = zdim

        self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
        #self.encoder = Encoder(self.cell, num_encoder_symbols=self.vocab_size, embedding_size=self.embedding_dim)


    def return_placeholders(self):
        encoder_inputs = list()
        decoder_inputs = list()
        labels = list()
        weights = list()

        #print(self.enc_seq_length)
        for _ in range(self.enc_seq_length):
            encoder_inputs.append(tf.placeholder(tf.int32, shape=(None, )))
        for _ in range(self.dec_seq_length):
            decoder_inputs.append(tf.placeholder(tf.int32, shape=(None,)))
            labels.append(tf.placeholder(tf.int32, shape=(None,)))
            weights.append(tf.placeholder(tf.float32, shape=(None, )))

        return encoder_inputs, decoder_inputs, labels, weights

    def output(self, encoder_inputs, decoder_inputs, z, generate_flag=False):
        # encode
        self.encoder = Encoder(self.vocab_size, self.batch_size, self.embedding_dim, self.hidden_size, self.enc_seq_length, self.start_token)
        h, c = self.encoder(encoder_inputs, generate_flag)

        # decode
        self.decoder = Decoder(self.vocab_size, self.batch_size, self.embedding_dim, self.hidden_size*2, self.dec_seq_length, self.start_token, self.encoder.lstm.embeddings)
        h = tf.concat([h, z], 1)
        c = tf.concat([c, z], 1)
        generateds, predictions, outputs, _ = self.decoder(c, h, decoder_inputs, generate_flag=generate_flag)
        return generateds, predictions, outputs

    def pretrain_loss(self, preds, labels, weights):
        loss = -tf.reduce_sum(
                tf.one_hot(tf.to_int32(tf.reshape(labels, [-1])), self.vocab_size, 1.0, 0.0) * tf.log(
                tf.clip_by_value(tf.reshape(preds, [-1, self.vocab_size]), 1e-20, 1.0))
                ) / (self.dec_seq_length * self.batch_size)
        # loss = tf.contrib.legacy_seq2seq.sequence_loss(preds, labels, weights)
        return loss

    def pretrain_opt(self, loss):
        params = self.encoder.lstm.params + self.decoder.lstm.params
        pretrain_opt = tf.train.AdamOptimizer(self.learning_rate)
        pretrain_grad, _ = tf.clip_by_global_norm(tf.gradients(loss, params), 5.0)
        pretrain_updates = pretrain_opt.apply_gradients(zip(pretrain_grad, params))
        return pretrain_updates
        #return tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)

    def train(self, preds, labels, weights, rewards):
        params = self.encoder.lstm.params + self.decoder.lstm.params
        loss = -tf.reduce_sum(
                tf.reduce_sum(tf.one_hot(tf.to_int32(tf.reshape(labels, [-1])), self.vocab_size, 1.0, 0.0) * tf.log(
                tf.clip_by_value(tf.reshape(preds, [-1, self.vocab_size]), 1e-20, 1.0)), 1) * tf.reshape(rewards, [-1]))

        loss = loss / (self.dec_seq_length * self.batch_size)
        opt = tf.train.AdamOptimizer(self.learning_rate)
        grad, _ = tf.clip_by_global_norm(tf.gradients(loss, params), 5.0)

        updates = opt.apply_gradients(zip(grad, params))

        return loss, updates

    def test_loss(self, predictions, labels, weights):
        loss = -tf.reduce_sum(
                tf.one_hot(tf.to_int32(tf.reshape(labels, [-1])), self.vocab_size, 1.0, 0.0) * tf.log(
                tf.clip_by_value(tf.reshape(predictions, [-1, self.vocab_size]), 1e-20, 1.0))
                ) / (self.dec_seq_length * self.batch_size)
        return loss

    def test(self, encoder_inputs, decoder_inputs, labels, z, generate_flag=True):
        # encode
        #self.encoder = Encoder(self.vocab_size, self.batch_size, self.embedding_dim, self.hidden_size, self.enc_seq_length, self.start_token)
        #h, c = self.encoder(encoder_inputs, generate_flag)

        # decode
        #self.decoder = Decoder(self.vocab_size, self.batch_size, self.embedding_dim, self.hidden_size, self.dec_seq_length, self.start_token, self.encoder.lstm.embeddings)
        #generateds, predictions, outputs, _ = self.decoder(c, h, decoder_inputs, generate_flag=generate_flag)
        generateds, predictions, outputs = self.output(encoder_inputs, decoder_inputs, z, generate_flag)

        #loss = -tf.reduce_sum(
        #        tf.one_hot(tf.to_int32(tf.reshape(labels, [-1])), self.vocab_size, 1.0, 0.0) * tf.log(
        #        tf.clip_by_value(tf.reshape(predictions, [-1, self.vocab_size]), 1e-20, 1.0))
        #        ) / (self.dec_seq_length * self.batch_size)

        return generateds
