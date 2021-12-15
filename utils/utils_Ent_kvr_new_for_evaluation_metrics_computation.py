import json
import ast

from utils.utils_general_for_kvr_evaluation_metrics_computation import *


def read_langs(file_name, lang, task, max_line=None):
    print(("Reading lines from {}".format(file_name)))
    data, context_arr, conv_arr, kb_arr, conv_arr_plain, kb_arr_plain = [], [], [], [], [], []
    max_resp_len = 0

    with open('data/KVR/kvret_entities.json') as f:
        global_entity = json.load(f)

    with open(file_name) as fin:
        cnt_lin, sample_counter, turn_cnt = 0, 0, 1
        for line in fin:
            line = line.strip()
            if line:
                if line.startswith("#"):
                    line = line.replace("#", "")
                    task_type = line
                    continue

                nid, line = line.split(' ', 1)
                if '\t' in line:
                    # deal with dialogue history
                    u, r, gold_ent = line.split('\t')
                    gen_u = generate_memory(u, "$u", str(nid))
                    context_arr += gen_u
                    conv_arr += gen_u
                    conv_arr_plain.append(u)

                    # Get gold entity for each domain
                    gold_ent = ast.literal_eval(gold_ent)
                    ent_idx_cal, ent_idx_nav, ent_idx_wet = [], [], []
                    if task_type == "weather": ent_idx_wet = gold_ent
                    elif task_type == "schedule": ent_idx_cal = gold_ent
                    elif task_type == "navigate": ent_idx_nav = gold_ent
                    ent_index = list(set(ent_idx_cal + ent_idx_nav + ent_idx_wet))

                    # Get local pointer position for each word in system response
                    ptr_index = []
                    for key in r.split():
                        index = [loc for loc, val in enumerate(kb_arr_plain) if (val == key and key in ent_index)]
                        if (index):
                            index = max(index)
                        else:
                            index = len(kb_arr_plain)
                        ptr_index.append(index)

                    # Get global pointer labels for words in system response, the 1 in the end is for the NULL token
                    selector_index = [1 if (word_arr in ent_index or word_arr in r.split()) else 0 for word_arr in
                                      kb_arr_plain] + [1]

                    # obtain gt entity labels
                    if len(gold_ent) == 0:
                        ent_labels = len(kb_arr_plain)
                    elif len(gold_ent) >= 1:
                        for idx, ent in enumerate(kb_arr_plain):
                            if ent in gold_ent:
                                ent_labels = idx
                                break

                    data_detail = {
                        'context_arr': list(context_arr + [['$$$$'] * MEM_TOKEN_SIZE]),  # $$$$ is NULL token
                        'response': r,
                        'ptr_index': ptr_index + [len(kb_arr_plain)],
                        'selector_index': selector_index,
                        'ent_index':ent_index,
                        'ent_idx_cal':list(set(ent_idx_cal)),
                        'ent_idx_nav':list(set(ent_idx_nav)),
                        'ent_idx_wet':list(set(ent_idx_wet)),
                        'conv_arr': list(conv_arr),
                        'conv_arr_plain': list(conv_arr_plain),
                        'kb_arr': list([['$$$$'] * MEM_TOKEN_SIZE]),
                        'sample_id': int(sample_counter),
                        'turn_id': int(turn_cnt),
                        'domain': task_type,
                        'kb_arr_plain': list(kb_arr_plain + ["[NULL]"]),
                        'ent_labels': 0,
                        'kb_arr_new': list(kb_arr_plain + ["[NULL]"]),
                    }
                    data.append(data_detail)

                    gen_r = generate_memory(r, "$s", str(nid))
                    context_arr += gen_r
                    conv_arr += gen_r
                    conv_arr_plain.append(r)
                    if max_resp_len < len(r.split()):
                        max_resp_len = len(r.split())
                    turn_cnt += 1
                else:
                    # deal with knowledge graph
                    r = line
                    line_list = line.split(" ")
                    if len(line_list) < 3:
                        continue
                    if line_list[0] not in kb_arr_plain:
                        kb_arr_plain.append(line_list[0])
                    if line_list[2] not in kb_arr_plain:
                        kb_arr_plain.append(line_list[2])
                    kb_info = generate_memory(r, "", str(nid))
                    # context_arr = kb_info + context_arr
                    kb_arr += kb_info
            else:
                cnt_lin += 1
                sample_counter += 1
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


def prepare_data_seq(task, batch_size=100):
    file_train = '/home/shiquan/Projects/tmp/GLMP/data/KVR/train_modified.txt'
    file_dev = '/home/shiquan/Projects/tmp/GLMP/data/KVR/dev_modified.txt'
    file_test = '/home/shiquan/Projects/tmp/GLMP/data/KVR/test_modified.txt'

    lang = Lang()

    pair_train, train_max_len = read_langs(file_train, lang, 'train', max_line=None)
    pair_dev, dev_max_len = read_langs(file_dev, lang, 'dev', max_line=None)
    pair_test, test_max_len = read_langs(file_test, lang, 'test', max_line=None)
    max_resp_len = max(train_max_len, dev_max_len, test_max_len) + 1

    train = get_seq(pair_train, lang, batch_size, True)
    dev = get_seq(pair_dev, lang, batch_size, False)
    test = get_seq(pair_test, lang, batch_size, False)

    print("Read %s sentence pairs train" % len(pair_train))
    print("Read %s sentence pairs dev" % len(pair_dev))
    print("Read %s sentence pairs test" % len(pair_test))
    print("Vocab_size: %s " % lang.n_words)
    print("Max. length of system response: %s " % max_resp_len)
    print("USE_CUDA={}".format(USE_CUDA))

    return train, dev, test, [], lang, max_resp_len


def get_data_seq(file_name, lang, max_len, batch_size=1):
    pair, _ = read_langs(file_name, max_line=None)
    # print(pair)
    d = get_seq(pair, lang, batch_size, False)
    return d
