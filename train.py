# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf
import time
import copy

# 人様のコード
from discriminator import Discriminator
#from rollout import ROLLOUT
# 自分のコード
#from model import Seq2Seq
from model import Seq2Seq
from rollout import Rollout
#from rollout import Rollout
from squad import Squad  # text2id, データを生成する
from dataloader import Gen_Data_Loader
from dataloader import Dis_Data_Loader
from ops_train import *

##### path名 #####
SQUAD_DATA = '/home/nakanishi/data/squad/train-v1.1.json'
OUTPUT_DIR = '/home/nakanishi/my_works/gan/conditionalGan/save'
EVENTS_DIR = '/home/nakanishi/my_works/gan/conditionalGan/save/events'
in_train_data_filename = 'in_train.txt'
in_eval_data_filename = 'in_eval.txt'
out_train_data_filename = 'out_train.txt'
out_eval_data_filename = 'out_eval.txt'


##### 学習のパラメータ #####
PRE_EPOCH_NUM = 120  # 120
PRE_TIMES_NUM = 50  # discriminatorを3epoch pre-trainingする回数 50
TOTAL_BATCH = 20000  # 200, adversarial-training 回数

BATCH_SIZE = 64
EMBEDDING_DIM = 128
HIDDEN_SIZE = 130
ZDIM = HIDDEN_SIZE

START_TOKEN = 0
LEARNING_RATE = 0.01
GENERATED_NUM = 10000
UPDATE_RATE = 0.8
#  Discriminator  Hyper-parameters
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64

##### 使用するデータのパラメータ #####
USE_TRAINED_MODEL = False
TRAINED_MODEL_PATH = '/home/nakanishi/my_works/gan/conditionalGan/save/pretrain/17_1114_1109/'
SAVE_PRE_TRAIN = True #pretrainを保存するかしないか
DATA_RESET = False  # 新しくデータを生成するかどうか
IN_SEQ_LENGTH = 100
OUT_SEQ_LENGTH = 20
VOCAB_SIZE = 5000
MAX_DATA_NUM = 10000
GPU_NUMBER = "3"


def init_train():
    # generatorを学習するためのデータを構築
    build_file_names = [in_train_data_filename, in_eval_data_filename,
        out_train_data_filename, out_eval_data_filename]
    build_file_pathes = [os.path.join(
        OUTPUT_DIR, build_file_names[i]) for i in range(4)]
    Squad = make_datafile(build_file_pathes, DATA_RESET, OUTPUT_DIR, SQUAD_DATA,
                          VOCAB_SIZE, IN_SEQ_LENGTH, OUT_SEQ_LENGTH, MAX_DATA_NUM)

    # data_loaderを準備
    print('Loading Data...')
    gen_data_loader = Gen_Data_Loader(BATCH_SIZE)
    gen_test_data_loader = Gen_Data_Loader(BATCH_SIZE)

    dis_data_loader = Dis_Data_Loader(BATCH_SIZE, OUT_SEQ_LENGTH)

    # eventsフォルダをからにする
    if len(glob.glob(os.path.join(EVENTS_DIR, 'events*'))) > 0:
        os.remove(os.path.join(EVENTS_DIR, 'events*'))

    # 使用するGPUを指定
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(
        visible_device_list=GPU_NUMBER))
    return Squad, gen_data_loader, gen_test_data_loader, dis_data_loader, config

################################################################################
graph = tf.Graph()
with graph.as_default():

    generator = Seq2Seq(VOCAB_SIZE, BATCH_SIZE, embedding_dim=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE,
                        enc_seq_length=IN_SEQ_LENGTH, dec_seq_length=OUT_SEQ_LENGTH,
                        zdim=ZDIM, learning_rate=LEARNING_RATE)

    discriminator = Discriminator(sequence_length=OUT_SEQ_LENGTH, num_classes=2,
                                vocab_size=VOCAB_SIZE, embedding_size=dis_embedding_dim,
                                filter_sizes=dis_filter_sizes, num_filters=dis_num_filters,
                                l2_reg_lambda=dis_l2_reg_lambda)


    # generator placeholders
    encoder_inputs, decoder_inputs, labels, weights = generator.return_placeholders()
    z = tf.placeholder(tf.float32, [BATCH_SIZE, ZDIM])

    # encode-decode
    gens, preds, o = generator.output(encoder_inputs, decoder_inputs, z, generate_flag=False)

    # pretrain operations
    pre_loss_op = generator.pretrain_loss(preds, labels, weights)
    #tf.summary.scalar('pre-train_loss', pre_loss_op)
    pre_opt_op = generator.pretrain_opt(pre_loss_op)

    # adversarial operations
    rewards = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUT_SEQ_LENGTH]) # get from rollout policy and discriminator
    loss_op, opt_op = generator.train(preds, labels, weights, rewards)
    tf.summary.scalar('train_loss', loss_op)

    # test
    test_loss_op = generator.test_loss(preds, labels, weights)
    #tf.summary.scalar('test_loss', test_loss_op)
    test_gens =  generator.test(encoder_inputs, decoder_inputs, labels, z, generate_flag=True)
    #test_loss_op, test_gens =  generator.test(encoder_inputs, decoder_inputs, labels, generate_flag=False)

    # rollout
    rollout = Rollout(generator, UPDATE_RATE)
    ro_encoder_inputs, ro_decoder_inputs, ro_labels, ro_weights, ro_given_num = rollout.return_placeholders()
    rollout_gens = rollout.output(ro_encoder_inputs, ro_decoder_inputs, z, ro_given_num, generate_flag=True)

    #rollout_update_op = rollout.update()
    #rollout_loss_op = rollout.loss(ro_preds, ro_labels, ro_weights)
    #ro_opt_op = rollout.train_opt(rollout_loss_op)

    summary_op = tf.summary.merge_all()
################################################################################

def return_feed_dict(sess, X_batch, label_batch, y_batch, w_batch):
    feed_dict = {}
    feed_dict = {encoder_inputs[i]: X_batch[i] for i in range(IN_SEQ_LENGTH)}
    feed_dict.update({decoder_inputs[i]: y_batch[i]
                      for i in range(OUT_SEQ_LENGTH)})
    feed_dict.update({labels[i]: label_batch[i]
                              for i in range(OUT_SEQ_LENGTH)})  # 正解データ
    feed_dict.update({weights[i]: w_batch[i] for i in range(OUT_SEQ_LENGTH)})

    feed_dict.update({z: np.random.randn(BATCH_SIZE, ZDIM)})
    return feed_dict


def generate(sess, gen_data_loader, num_samples, output_path=None):
    generated_samples = []
    for _ in range(int(num_samples/gen_data_loader.batch_size)):
        X_batch, label_batch, y_batch, w_batch = gen_data_loader.next_batch()
        feed_dict = return_feed_dict(sess, X_batch, label_batch, y_batch, w_batch)
        #feed_dict.update({z: np.random.randn(BATCH_SIZE, ZDIM)})
        #feed_dict.update({c:encoder_c, h:encoder_h, at:attention_states})

        generateds = sess.run(gens, feed_dict=feed_dict) # (batch_size, seq_length)
        generated_samples.extend(generateds)

    if output_path != None:
        save_generated(generated_samples, output_path)
    return generated_samples


def get_reward(sess, generated_samples, gen_data_loader, rollout_num):
    rewards = []
    for i in range(rollout_num):
        for given_num in range(1, 20):
            X_batch, _, y_batch, w_batch = gen_data_loader.next_batch()
            label_batch = np.array(generated_samples).T # batch_size, dec_seq_length
            y_batch = []
            for line in generated_samples:
                line = [0] + line[:len(line)]
                y_batch.append(line)
            y_batch = np.array(y_batch).T
            #feed_dict = return_feed_dict(sess, X_batch, label_batch, y_batch, w_batch, rollout=True)

            feed_dict = {}
            feed_dict = {ro_encoder_inputs[i]: X_batch[i] for i in range(IN_SEQ_LENGTH)}
            feed_dict.update({ro_decoder_inputs[i]: y_batch[i] for i in range(OUT_SEQ_LENGTH)})
            feed_dict.update({ro_labels[i]: label_batch[i] for i in range(OUT_SEQ_LENGTH)})  # 正解データ
            feed_dict.update({ro_weights[i]: w_batch[i] for i in range(OUT_SEQ_LENGTH)})
            feed_dict.update({z: np.random.randn(BATCH_SIZE, ZDIM)})
            feed_dict.update({ro_given_num:given_num})

            samples = sess.run(rollout_gens, feed_dict=feed_dict)

            feed = {discriminator.input_x: samples,
                    discriminator.dropout_keep_prob: 1.0}
            ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
            ypred = np.array([item[1] for item in ypred_for_auc])
            if i == 0:
                rewards.append(ypred)
            else:
                rewards[given_num - 1] += ypred

        feed = {discriminator.input_x: generated_samples,
                discriminator.dropout_keep_prob: 1.0}
        ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
        yred = np.array([item[1] for item in ypred_for_auc])
        if i == 0:
            rewards.append(ypred)
        else:
            rewards[19] += ypred
    rewards = np.transpose(np.array(rewards)) / (1.0 * rollout_num) # (batch_size, dec_seq_length)
    return rewards


def train_discriminator(sess, dis_data_loader, ti, positive_file_path, negative_file_path):
    # データをdiscriminator_data_loaderへセット
    dis_data_loader.load_train_data(positive_file_path, negative_file_path)

    dis_losses = []
    for epoch in range(3):
        dis_data_loader.reset_pointer()
        for it in range(dis_data_loader.num_batch):
            x_batch, y_batch = dis_data_loader.next_batch()
            feed = {
                discriminator.input_x: x_batch,
                discriminator.input_y: y_batch,
                discriminator.dropout_keep_prob: dis_dropout_keep_prob
            }
            _, dis_loss = sess.run(
                [discriminator.train_op, discriminator.loss], feed)
            dis_losses.append(dis_loss)
    dis_loss = np.mean(dis_losses)
    print('times ', ti, ' loss ', dis_loss)

    return dis_loss


def pretrain(sess, gen_data_loader, epoch, Squad, Test=False):
    gen_data_loader.reset_pointer()
    # train 1 epoch
    loss = []
    for it in range(gen_data_loader.num_batch):
        X_batch, label_batch, y_batch, w_batch = gen_data_loader.next_batch()
        feed_dict = return_feed_dict(sess, X_batch, label_batch, y_batch, w_batch)

        if not Test:
            generateds, predictions, l = sess.run([gens, preds, pre_loss_op], feed_dict=feed_dict)
            opt = sess.run(pre_opt_op, feed_dict=feed_dict)
        if Test:
            #generateds, l = sess.run([test_gens, test_loss_op], feed_dict=feed_dict)
            generateds, predictions, l = sess.run([gens, preds, pre_loss_op], feed_dict=feed_dict)

        loss.append(l)
    loss = np.mean(loss)

    # prediction保存
    if not Test:
        print('epoch ', epoch, ' loss ', loss)
        if (epoch + 1) % 5 == 0:
            prediction_path = os.path.join(
                RESULT_DIR, 'pre-train_prediction_{0}.txt'.format(('{0:03d}'.format(epoch + 1))))
            save_predictions(
                generateds, X_batch[:IN_SEQ_LENGTH], label_batch[:OUT_SEQ_LENGTH], prediction_path, Squad)
            #save_genx(generateds, label_batch, prediction_path, Squad)
    if Test:
        print('Test loss', loss)
        prediction_path = os.path.join(RESULT_DIR, 'pre-test_prediction_{0}.txt'.format(('{0:03d}'.format(epoch+1))))
        save_predictions(generateds, X_batch[:IN_SEQ_LENGTH], label_batch[:OUT_SEQ_LENGTH], prediction_path, Squad)

    return loss

def train(sess, gen_data_loader, total_batch, Squad):
    generated_samples = generate(sess, gen_data_loader, BATCH_SIZE, None) # 1batch分生成
    rwd = get_reward(sess, generated_samples, gen_data_loader, 16)
    X_batch, _, _, w_batch = gen_data_loader.next_batch()
    label_batch = np.array(generated_samples).T # batch_size, dec_seq_length
    y_batch = []
    for line in generated_samples:
        line = [0] + line[:len(line)]
        y_batch.append(line)
    y_batch = np.array(y_batch).T
    feed_dict = return_feed_dict(sess, X_batch, label_batch, y_batch, w_batch)
    feed_dict.update({rewards: rwd})
    loss, opt, summary = sess.run([loss_op, opt_op, summary_op], feed_dict=feed_dict)
    print('Generator loss ', loss)

    if (total_batch+1) % 2 == 0:
        X_batch, label_batch, y_batch, w_batch = gen_data_loader.next_batch()
        feed_dict = return_feed_dict(sess, X_batch, label_batch, y_batch, w_batch)

        #generateds = sess.run(test_gens, feed_dict=feed_dict)

        prediction_path = os.path.join(RESULT_DIR, 'train_prediction_{0}.txt'.format('{0:03d}'.format(total_batch+1)))
        save_predictions(generated_samples, X_batch[:IN_SEQ_LENGTH], label_batch[:OUT_SEQ_LENGTH], prediction_path, Squad)

    return loss, summary



def test(sess, data_loader, total_batch, Squad):
    # 全テスト用データに対してテスト
    data_loader.reset_pointer()
    loss = []
    for it in range(data_loader.num_batch):
        X_batch, label_batch, y_batch, w_batch = data_loader.next_batch()
        feed_dict = return_feed_dict(sess, X_batch, label_batch, y_batch, w_batch)
        generateds, predictions, l = sess.run([gens, preds, test_loss_op], feed_dict=feed_dict)
        loss.append(l)
    loss = np.mean(loss)
    test_generateds = sess.run(test_gens, feed_dict=feed_dict)
    print('Test loss ', loss)
    prediction_path = os.path.join(RESULT_DIR, 'test_prediction_{0}.txt'.format(('{0:03}'.format(total_batch+1))))
    save_predictions(generateds, X_batch[:IN_SEQ_LENGTH], label_batch[:OUT_SEQ_LENGTH], prediction_path, Squad)

    prediction_path = os.path.join(RESULT_DIR, 'generateds_test.txt')
    save_predictions(test_generateds, None, None, prediction_path, Squad)
    return loss



def main(RESULT_DIR, TIME_NAME):
    Squad, gen_data_loader, gen_test_data_loader, dis_data_loader, config = init_train()

    start = time.time()
    ADVERSAL = False

    with tf.Session(graph=graph, config=config) as sess:
        sess.run(tf.global_variables_initializer())
        # TODO: バッチ生成の時にランダムに並ぶように変更して, 何回かに1回バッチ生成する
        gen_data_loader.create_batches(
            Squad.in_train_data_file, Squad.out_train_data_file, IN_SEQ_LENGTH, OUT_SEQ_LENGTH)
        gen_test_data_loader.create_batches(
            Squad.in_eval_data_file, Squad.out_eval_data_file, IN_SEQ_LENGTH, OUT_SEQ_LENGTH)
        # log file
        log = open(os.path.join(RESULT_DIR, 'log.txt'), 'w')
        log = log_init(log, gen_data_loader, Squad)
        saver = tf.train.Saver()

        writer = tf.summary.FileWriter(EVENTS_DIR, sess.graph)
        ############# pretrain #####################
        if USE_TRAINED_MODEL:
            ckpt = tf.train.get_checkpoint_state(TRAINED_MODEL_PATH)
            if ckpt:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('restore ', ckpt.model_checkpoint_path)
                log.write('RESTORE MODEL, '+ str(ckpt)+'\n')
            else:
                print('there is not ckpt model in ', USE_PRE_TRAIN_MODEL)
                sys.exit()
        else:
            try:
                print('\n===== Start Pre-Training =====')
                print('Generator')
                log.write('----- pre Training Generator -----\n')

                for epoch in range(PRE_EPOCH_NUM):
                    loss = pretrain(sess, gen_data_loader, epoch, Squad)
                    log.write('epoch '+str(epoch)+' loss '+str(loss) + '\n')
                    if (epoch+1) % 5 == 0:
                        loss = pretrain(sess,gen_test_data_loader, epoch, Squad, Test=True)
                        log.write('TEST loss'+str(loss)+'\n')

                print('Discriminator')
                log.write('----- pre Training Discriminator -----\n')
                for ti in range(PRE_TIMES_NUM):  # 本当は50
                    gen_data_loader.reset_pointer()
                    # 全データ数分のgenerated samplesを生成
                    generated_samples = generate(sess, gen_data_loader, gen_data_loader.num_samples,
                                                            os.path.join(OUTPUT_DIR, 'generated.txt'))

                    dis_loss = train_discriminator(sess, dis_data_loader, ti, Squad.out_train_data_file, os.path.join(OUTPUT_DIR,'generated.txt'))
                    log.write('times ' + str(ti) + ' loss ' + str(dis_loss) + '\n')

                # pre-train モデルの保存
                if SAVE_PRE_TRAIN:
                    os.mkdir(os.path.join(OUTPUT_DIR, 'pretrain', TIME_NAME))
                    pre_model_path = os.path.join(OUTPUT_DIR, 'pretrain', TIME_NAME,'pre_model.ckpt')
                    saver.save(sess, pre_model_path, global_step=epoch)
                    print('save model ', pre_model_path)
            except Exception as e:
                print('===== エラー内容 =====')
                import traceback
                traceback.print_exc()
                print('====================')
                if SAVE_PRE_TRAIN:
                    os.mkdir(os.path.join(OUTPUT_DIR, 'pretrain', TIME_NAME))
                    pre_model_path = os.path.join(OUTPUT_DIR, 'pretrain', TIME_NAME,'pre_model.ckpt')
                    saver.save(sess, pre_model_path, global_step=epoch)
                    print('save model ', pre_model_path)
                log.close()


        ############# adversarial #####################
        print('\n===== Start Adversarial Training =====')
        log.write('----- Adversarial Training -----\n')
        try:
            for total_batch in range(TOTAL_BATCH):
                print('Batch ', total_batch)
                # train generator for 1 batch (1step)
                for it in range(1):
                    loss, summary = train(sess, gen_data_loader, total_batch, Squad)
                    log.write('batch ' + str(total_batch) + '\n' )
                    log.write('Generator        :'+str(loss)+'\n')
                # update rollout parameters
                rollout.update()
                writer.add_summary(summary, total_batch)

                # trian discriminator
                print('Discriminator')
                for it in range(5):
                    dis_losses = []
                    generated_samples = generate(sess, gen_data_loader, gen_data_loader.num_samples,
                                                            os.path.join(OUTPUT_DIR, 'generated.txt'))
                    dis_loss = train_discriminator(sess, dis_data_loader, it, Squad.out_train_data_file, os.path.join(OUTPUT_DIR,'generated.txt'))
                    dis_losses.append(dis_loss)
                log.write('Discriminator    :'+str(np.mean(dis_losses))+'\n')

                if (total_batch+1) % 5 == 0:
                    loss = test(sess, gen_data_loader, total_batch, Squad)
                    log.write('Test loss '+str(loss)+'\n')
                if (total_batch+1) % 20 == 0:
                    model_path = os.path.join(RESULT_DIR, 'model.ckpt')
                    saver.save(sess, model_path, global_step=total_batch)
                    print('save model ', model_path)

        except Exception as e:
            print('===== エラー内容 =====')
            import traceback
            traceback.print_exc()
            print('====================')

            log.close()
            model_path = os.path.join(RESULT_DIR, 'model.ckpt')
            saver.save(sess, model_path, global_step=total_batch)
            print('save model ', model_path)

if __name__ == '__main__':
    TIME_NAME = return_time_name()
    RESULT_DIR = os.path.join(OUTPUT_DIR, TIME_NAME)
    os.mkdir(RESULT_DIR)
    print('\nMake Directory ', RESULT_DIR)

    main(RESULT_DIR, TIME_NAME)
