import json
import ast
import pandas as pd
import random

from utils.utils_general_for_entity_prediction import *


def read_langs(file_name, lang, task, max_line=None):
    print(("Reading lines from {}".format(file_name)))
    data, context_arr, conv_arr, kb_arr, conv_arr_plain, kb_arr_plain = [], [], [], [], [], []
    max_resp_len = 0

    generated_response_file = "/home/shiquan/Projects/tmp/GLMP/outputs/test_generated_responses_{}.json".format(args['dataset'])
    with open(generated_response_file, "r") as f:
        generated_responses = json.load(f)

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
                    # deal with dialogue history
                    u, r, gold_ent = line.split('\t')
                    gen_u = generate_memory(u, "$u", str(nid))
                    gen_r = generate_memory(r, "$s", str(nid))
                    if task == 'test':
                        generated_response = generated_responses[str(sample_counter)][str(turn_cnt)]['generated_response']
                        gen_r_new = generate_memory(generated_response, "$s", str(nid))
                        gold_ent = ast.literal_eval(gold_ent)
                        context_arr += gen_u
                        r_list = generated_response.split(" ")
                        for idx, token in enumerate(r_list):
                            # inputs = context_arr + gen_r_new[:idx]
                            inputs = context_arr
                            if token in gold_ent:
                                target = token
                            else:
                                target = "NULL"
                            data_detail = {
                                'input': inputs,
                                'target': target,
                                'sample_id': int(sample_counter),
                                'turn_cnt': int(turn_cnt),
                                'timestep': int(idx)
                            }
                            data.append(data_detail)

                        if max_resp_len < len(r.split()):
                            max_resp_len = len(r.split())
                        context_arr += gen_r
                        turn_cnt += 1
                    else:
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
                                'target': target,
                                'sample_id': int(sample_counter),
                                'turn_cnt': int(turn_cnt),
                                'timestep': int(idx)
                            }
                            data.append(data_detail)

                        if max_resp_len < len(r.split()):
                            max_resp_len = len(r.split())
                        context_arr += gen_r
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
                    context_arr = kb_info + context_arr
                    kb_arr += kb_info
            else:
                cnt_lin += 1
                turn_cnt = 1
                sample_counter += 1
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


def read_langs_multiwoz(file_name, lang, task, max_line=None):
    print(("Reading lines from {}".format(file_name)))
    data, context_arr, conv_arr, kb_arr, conv_arr_plain, kb_arr_plain = [], [], [], [], [], []
    max_resp_len = 0

    generated_response_file = "/home/shiquan/Projects/tmp/GLMP/outputs/test_generated_responses_{}.json".format(args['dataset'])
    with open(generated_response_file, "r") as f:
        generated_responses = json.load(f)

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
                    # deal with dialogue history
                    u, r, gold_ent = line.split('\t')
                    gen_u = generate_memory(u, "$u", str(nid))
                    gen_r = generate_memory(r, "$s", str(nid))
                    if task == 'test':
                        generated_response = generated_responses[str(sample_counter)][str(turn_cnt)]['generated_response']
                        gen_r_new = generate_memory(generated_response, "$s", str(nid))
                        gold_ent = ast.literal_eval(gold_ent)
                        context_arr += gen_u
                        r_list = generated_response.split(" ")
                        for idx, token in enumerate(r_list):
                            # inputs = context_arr + gen_r_new[:idx]
                            inputs = context_arr
                            if token in gold_ent:
                                target = token
                            else:
                                target = "NULL"
                            data_detail = {
                                'input': inputs,
                                'target': target,
                                'sample_id': int(sample_counter),
                                'turn_cnt': int(turn_cnt),
                                'timestep': int(idx)
                            }
                            data.append(data_detail)

                        if max_resp_len < len(r.split()):
                            max_resp_len = len(r.split())
                        context_arr += gen_r
                        turn_cnt += 1
                    else:
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
                                'target': target,
                                'sample_id': int(sample_counter),
                                'turn_cnt': int(turn_cnt),
                                'timestep': int(idx)
                            }
                            data.append(data_detail)

                        if max_resp_len < len(r.split()):
                            max_resp_len = len(r.split())
                        context_arr += gen_r
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
                    context_arr = kb_info + context_arr
                    kb_arr += kb_info
            else:
                cnt_lin += 1
                turn_cnt = 1
                sample_counter += 1
                context_arr, conv_arr, kb_arr, conv_arr_plain, kb_arr_plain = [], [], [], [], []
                if (max_line and cnt_lin >= max_line):
                    break

    return data, max_resp_len


def prepare_data_seq(task, batch_size=100):
    # file_train = '/Users/shiquan/PycharmProjects/GLMP/data/multiwoz/train_utterances_w_kb_w_gold_w_bias_p=0_5_sm.txt'
    # file_dev = '/Users/shiquan/PycharmProjects/GLMP/data/multiwoz/dev_utterances_w_kb_w_gold_w_bias_p=0_5_sm.txt'
    # file_test = '/Users/shiquan/PycharmProjects/GLMP/data/multiwoz/test_utterances_w_kb_w_gold_w_bias_p=0_5_sm.txt'
    file_train = '/home/shiquan/Projects/tmp/GLMP/data/multiwoz/train_utterances_w_kb_w_gold_leave_out_restaurant_removing_high_pmi_sevengram_tokens_for_all_labels.txt'
    file_dev = '/home/shiquan/Projects/tmp/GLMP/data/multiwoz/dev_utterances_w_kb_w_gold_leave_out_restaurant_removing_high_pmi_sevengram_tokens_for_all_labels.txt'
    file_test = '/home/shiquan/Projects/tmp/GLMP/data/multiwoz/test_utterances_w_kb_w_gold_only_restaurant.txt'

    lang = Lang()
    pair_train, train_max_len = read_langs_multiwoz(file_train, lang, 'train', max_line=None)
    train = get_seq(pair_train, lang, batch_size, True)

    file_train = '/home/shiquan/Projects/tmp/GLMP/data/KVR/train_modified.txt'
    file_dev = '/home/shiquan/Projects/tmp/GLMP/data/KVR/dev_modified.txt'
    file_test = '/home/shiquan/Projects/tmp/GLMP/data/KVR/test_modified.txt'

    pair_train, train_max_len = read_langs(file_train, lang, 'train', max_line=None)
    pair_dev, dev_max_len = read_langs(file_dev, lang, 'dev', max_line=None)
    pair_test, test_max_len = read_langs(file_test, lang, 'test', max_line=None)
    max_resp_len = max(train_max_len, dev_max_len, test_max_len) + 1

    train = get_seq(pair_train, lang, batch_size, False)
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
    d = get_seq(pair, lang, batch_size, False)
    return d


if __name__ == "__main__":
    prepare_data_seq('all')
