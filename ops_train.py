"""
trainで使う雑用関数達
"""
import os
import time
import datetime
import numpy as np
import json
import pickle

from squad import Squad  # text2id, データを生成する

def log_init(log, gen_data_loader, Squad):
    buffer = 'num samples   :'+str(gen_data_loader.num_samples) + '\n'
    buffer += 'vocab size   :'+str(Squad.vocab_size) + '\n'
    buffer += 'batch size   :'+str(gen_data_loader.batch_size) + '\n'
    buffer += 'num batch    :'+str(gen_data_loader.num_batch) + '\n'
    buffer += 'in sequence length   :'+str(Squad.in_seq_length) + '\n'
    buffer += 'out sequence length  :'+str(Squad.out_seq_length) + '\n'
    buffer += '\n'

    log.write(buffer)
    return log

def return_time_name():
    """
    日時ファイル名を返す
    """
    d = datetime.datetime.today()
    year = str(d.year)[2:]
    t = d.strftime("%m%d_%H%M")
    TIME_NAME = year + '_' + str(t)
    return TIME_NAME

def print_time(start):
    now = time.time()
    h = int((now - start) / (60*60))
    m = int((now - start) / 60)
    s = int((now-start) % 60)
    print('{0}[h]{1}[m]{2}[s]'.format(h, m, s))

def save_generated(generateds, filepath):
    """
    discriminatorのために, generatorが生成したものをnegative_fileとして保存.
    discriminatorはこれとpositive_fileを読み込む
    """
    # generateds (64, 20)
    with open(filepath, 'w')as f:
        for generated in generateds:
            for index in generated:
                f.write(str(index))
                f.write(' ')
            f.write('\n')
    #print('save ', filepath)

def save_genx(genx, labels, filepath, S):
    labels = list(np.array(labels).T) #(batch_size, seq_length)
    with open(filepath, 'w') as f:
        for poem, label in zip(genx, labels):
            #words = [str(S.return_word(int(x))) for x in poem]
                #print(words)
            f.write('-'*30)
            f.write('\n')
            buffer = ' '.join([str(S.return_word(int(l))) for l in label]) + '\n'
            f.write(buffer)
            buffer = ' '.join([str(S.return_word(int(x))) for x in poem]) + '\n'
            f.write(buffer)


def save_predictions(predictions, Xs, ys, filepath, S):
    """
    generatorが生成した質問を単語に変換して保存
    同時にparagraphとgold_questionを保存する.
    prediction; generator予測結果, Xs: その時のgenerator入力, ys: その時のlabel
    S: Squadオブジェクト
    """
    #predictions = list(np.array(predictions).transpose(1, 0, 2))
    # prediction: [64, 20]
    if Xs != None and ys != None:
        Xs = list(np.array(Xs).T) #(batch_size, seq_length)
        ys = list(np.array(ys).T)

        with open(filepath, 'w') as f:
            f.write('='*30+'\n')
            for line, X, y in zip(predictions, Xs, ys):
                f.write('-'*10+'input'+'-'*10+'\n')
                for w in X:
                    index = int(w)
                    word = S.return_word(index, error_word='***',output=False)
                    f.write(word)
                    f.write(' ')
                f.write('\n')
                f.write('-'*10+'label'+'-'*10+'\n')
                for w in y:
                    index = int(w)
                    word = S.return_word(index, error_word='***',output=True)
                    f.write(word)
                    f.write(' ')
                f.write('\n')
                f.write('-'*10+'prediction'+'-'*10+'\n')

                buffer = ' '.join([str(S.return_word(int(x))) for x in line]) + '\n'
                f.write(buffer)
                f.write('\n')

    else:
        with open(filepath, 'w') as f:
            for line in predictions:
                buffer = ' '.join([str(S.return_word(int(x))) for x in line]) + '\n'
                f.write(buffer)
                f.write('\n')
    print('save ', filepath)



def make_datafile(datafile_names, DATA_RESET, OUTPUT_DIR, SQUAD_DATA, VOCAB_SIZE, IN_SEQ_LENGTH,
                    OUT_SEQ_LENGTH,MAX_DATA_NUM):
    DAT_NAME = 'squad.dat'
    in_train_data_file, in_eval_data_file, out_train_data_file, out_eval_data_file = datafile_names

    if not DATA_RESET:
        if os.path.exists(in_train_data_file) and os.path.exists(in_eval_data_file):
            if os.path.exists(os.path.join(OUTPUT_DIR, DAT_NAME)):
                print('Exists Datafiles Already.')
                with open(os.path.join(OUTPUT_DIR, DAT_NAME), 'rb') as f:
                    return pickle.load(f)

    print('Building Datafiles...')

    # input data (encode data)
    data_path = SQUAD_DATA
    with open(data_path) as f:
        data = json.load(f)

        Loader = Squad(data, vocab_size=VOCAB_SIZE, in_seq_length=IN_SEQ_LENGTH,
                       out_seq_length=OUT_SEQ_LENGTH, max_data_num=MAX_DATA_NUM)

        Loader.save_textfile(in_train_data_file,
                             out_train_data_file, limit=0.8)
        Loader.save_textfile(in_eval_data_file, out_eval_data_file, limit=0.2, Train=False)

        with open(os.path.join(OUTPUT_DIR, DAT_NAME), 'wb') as f:
            pickle.dump(Loader, f)
        print('save ', (os.path.join(OUTPUT_DIR, DAT_NAME)))

    return Loader
