import json
import ast
import random

from utils.utils_general_for_contrastive_learning_new import *


def read_langs(file_name, lang, task, max_line=None):
    print(("Reading lines from {}".format(file_name)))
    data, context_arr, conv_arr, kb_arr, conv_arr_plain, kb_arr_plain = [], [], [], [], [], []
    max_resp_len = 0

    # with open('data/multiwoz/multiwoz_entities.json') as f:
    #     global_entity = json.load(f)

    # filtered_ids = pd.read_csv(
    #     '/home/shiquan/Projects/tmp/GLMP/data/multiwoz/aflite_filtered_ids_leave_out_restaurant_20000_50_0.9.csv')
    # df = pd.DataFrame(filtered_ids)
    # filtered_ids = df['sample_id'].tolist()

    with open(file_name) as fin:
        cnt_lin, sample_counter, turn_cnt = 1, 0, 1
        for line in fin:
            line = line.strip()
            if line:
                if line.startswith("#"):
                    line = line.replace("#", "")
                    task_type = line
                    continue

                nid, line = line.split(' ', 1)
                if '\t' in line:
                    # if sample_counter in filtered_ids and task == 'train':
                    # if random.random() < 0.25 and task == 'train':
                    #     sample_counter += 1
                    #     continue
                    # deal with dialogue history
                    u, r, gold_ent = line.split('\t')
                    gen_u = generate_memory(u, "$u", str(nid))
                    gen_r = generate_memory(r, "$s", str(nid))
                    gold_ent = ast.literal_eval(gold_ent)
                    context_arr += gen_u
                    r_list = r.split(" ")
                    for idx, token in enumerate(r_list):
                        inputs = context_arr + gen_r[:idx]
                        if token in gold_ent:
                            target = token
                        else:
                            target = "NULL"
                        data_detail = {
                            'input': inputs,
                            'target': target
                        }
                        data.append(data_detail)

                    if max_resp_len < len(r.split()):
                        max_resp_len = len(r.split())
                    context_arr += gen_r
                    sample_counter += 1
                    turn_cnt += 1
                else:
                    # deal with knowledge graph
                    r = line
                    line_list = line.split(" ")
                    if line_list[0] not in kb_arr_plain:
                        kb_arr_plain.append(line_list[0])
                    if line_list[2] not in kb_arr_plain:
                        kb_arr_plain.append(line_list[2])
                    kb_info = generate_memory(r, "", str(nid))
                    if len(kb_info[0]) > 4:
                        print(kb_info)
                        print(r)
                    context_arr = kb_info + context_arr
                    kb_arr += kb_info
            else:
                cnt_lin += 1
                turn_cnt = 1
                context_arr, conv_arr, kb_arr, conv_arr_plain, kb_arr_plain = [], [], [], [], []
                if (max_line and cnt_lin >= max_line):
                    break

    return data, max_resp_len


def generate_memory(sent, speaker, time):
    sent_new = []
    sent_token = sent.split(' ')
    if speaker == "$u" or speaker == "$s":
        for idx, word in enumerate(sent_token):
            temp = [word, speaker, 'turn' + str(time), 'word' + str(idx)] + ["PAD"] * (MEM_TOKEN_SIZE - 4)
            sent_new.append(temp)
    else:
        sent_token = sent_token[::-1] + ["PAD"] * (MEM_TOKEN_SIZE - len(sent_token))
        sent_new.append(sent_token)
    return sent_new


def read_langs_paired_examples(file_name, lang, task, max_line=None):
    print(("Reading lines from {}".format(file_name)))
    data = []
    max_resp_len = 0

    with open(file_name) as fin:
        for line in fin:
            line = line.strip()
            if line:
                if '\t' in line:
                    # deal with dialogue history
                    sent1, sent2 = line.split('\t')
                    sent1_list, sent2_list = sent1.split(" "), sent2.split(" ")
                    sent1_trunc = trunc_seq(sent1_list, 500)
                    sent2_trunc = trunc_seq(sent2_list, 500)
                    paired_example = [sent1_trunc, sent2_trunc]

                    data_detail = {
                        'paired_example': list(paired_example)
                    }
                    data.append(data_detail)
                else:
                    continue
            else:
                continue

    return data, max_resp_len


def trunc_seq(sent, max_seq_len):
    if len(sent) > max_seq_len:
        res = sent[:max_seq_len]
    else:
        res = sent
    return res


def prepare_data_seq(task, batch_size=100):
    file_train = '/Users/shiquan/PycharmProjects/GLMP/data/multiwoz/train_utterances_w_kb_w_gold_w_bias_p=0_5_sm.txt'
    file_dev = '/Users/shiquan/PycharmProjects/GLMP/data/multiwoz/dev_utterances_w_kb_w_gold_w_bias_p=0_5_sm.txt'
    file_test = '/Users/shiquan/PycharmProjects/GLMP/data/multiwoz/test_utterances_w_kb_w_gold_w_bias_p=0_5_sm.txt'
    # file_train = '/home/yimeng/shiquan/debiasing-glmp/GLMP/data/MultiWOZ_2.2/train/train_utterances_w_kb_w_gold_w_bias.txt'
    # file_dev = '/home/yimeng/shiquan/debiasing-glmp/GLMP/data/MultiWOZ_2.2/dev/dev_utterances_w_kb_w_gold.txt'
    # file_test = '/home/yimeng/shiquan/debiasing-glmp/GLMP/data/MultiWOZ_2.2/test/test_utterances_w_kb_w_gold.txt'

    lang = Lang()

    pair_train, train_max_len = read_langs(file_train, lang, 'train', max_line=None)
    pair_dev, dev_max_len = read_langs(file_dev, lang, 'dev', max_line=None)
    pair_test, test_max_len = read_langs(file_test, lang, 'test', max_line=None)
    max_resp_len = max(train_max_len, dev_max_len, test_max_len) + 1

    train = get_seq(pair_train, lang, batch_size, True)
    dev = get_seq(pair_dev, lang, batch_size, False)
    test = get_seq(pair_test, lang, batch_size, False)

    # load contrastive learning paired samples
    file_paired_examples = '/Users/shiquan/PycharmProjects/GLMP/data/multiwoz/paired_examples_for_contrastive_learning_delete_seven_grams_for_all_labels.txt'
    pair_train, train_max_len = read_langs_paired_examples(file_paired_examples, lang, 'train', max_line=None)
    train_cl = get_seq_cl(pair_train, lang, batch_size, False)

    print("Read %s sentence pairs train" % len(pair_train))
    print("Read %s sentence pairs dev" % len(pair_dev))
    print("Read %s sentence pairs test" % len(pair_test))
    print("Vocab_size: %s " % lang.n_words)
    print("Max. length of system response: %s " % max_resp_len)
    print("USE_CUDA={}".format(USE_CUDA))

    return train_cl, dev, test, [], lang, max_resp_len


def get_data_seq(file_name, lang, max_len, batch_size=1):
    pair, _ = read_langs(file_name, max_line=None)
    d = get_seq(pair, lang, batch_size, False)
    return d