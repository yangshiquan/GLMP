import json
import ast

from utils.utils_general_for_gpt_finetuning import *


def read_langs(file_name, lang, task, max_line=None):
    print(("Reading lines from {}".format(file_name)))
    data, context_arr, conv_arr, kb_arr, conv_arr_plain, kb_arr_plain = [], [], [], [], [], []
    max_resp_len = 0

    with open('data/multiwoz/multiwoz_entities.json') as f:
        global_entity = json.load(f)

    with open(file_name) as fin:
        cnt_lin, sample_counter, turn_cnt = 0, 1, 1
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
                    if turn_cnt == 1:
                        dialogue_history = "<|endoftext|> <|context|> <|user|> " + u
                    else:
                        dialogue_history = dialogue_history + " <|user|> " + u

                    gold_ent = ast.literal_eval(gold_ent)
                    sketch_response = generate_template(global_entity, r, gold_ent, kb_arr, task_type)
                    gpt_input = dialogue_history + " <|endofcontext|> <|response|> " + sketch_response + " <|endofresponse|> <|endoftext|>"

                    data_detail = {
                        'gpt_input': gpt_input,  # $$$$ is NULL token
                        'kb_arr_plain': list(kb_arr_plain)
                    }
                    data.append(data_detail)

                    dialogue_history = dialogue_history + " <|system|> " + r
                    if max_resp_len < len(r.split()):
                        max_resp_len = len(r.split())
                    sample_counter += 1
                    turn_cnt += 1
                else:
                    # deal with knowledge graph
                    line_list = line.split(" ")
                    if line_list[0] not in kb_arr_plain:
                        kb_arr_plain.append(line_list[0])
                    if line_list[1] not in kb_arr_plain:
                        kb_arr_plain.append(line_list[1])
                    if line_list[2] not in kb_arr_plain:
                        kb_arr_plain.append(line_list[2])
            else:
                cnt_lin += 1
                turn_cnt = 1
                kb_arr_plain = []
                if (max_line and cnt_lin >= max_line):
                    break

    return data, max_resp_len


def generate_template(global_entity, sentence, sent_ent, kb_arr, domain):
    """
    Based on the system response and the provided entity table, the output is the sketch response.
    """
    sketch_response = []
    if sent_ent == []:
        sketch_response = sentence.split()
    else:
        for word in sentence.split():
            if word not in sent_ent:
                sketch_response.append(word)
            else:
                ent_type = None
                for key in global_entity.keys():
                    global_entity[key] = [x.lower() for x in global_entity[key]]
                    if word in global_entity[key] or word.replace('_', ' ') in global_entity[key]:
                        ent_type = key
                        break
                sketch_response.append('@' + ent_type)
    sketch_response = " ".join(sketch_response)
    return sketch_response


def prepare_data_seq(task, batch_size=100):
    # file_train = '/Users/shiquan/PycharmProjects/GLMP/data/multiwoz/train_utterances_w_kb_w_gold_w_bias_p=0_5_sm.txt'
    # file_dev = '/Users/shiquan/PycharmProjects/GLMP/data/multiwoz/dev_utterances_w_kb_w_gold_w_bias_p=0_5_sm.txt'
    # file_test = '/Users/shiquan/PycharmProjects/GLMP/data/multiwoz/test_utterances_w_kb_w_gold_w_bias_p=0_5_sm.txt'
    file_train = '/home/shiquan/Projects/tmp/GLMP/data/multiwoz/train_utterances_w_kb_w_gold.txt'
    file_dev = '/home/shiquan/Projects/tmp/GLMP/data/multiwoz/dev_utterances_w_kb_w_gold.txt'
    file_test = '/home/shiquan/Projects/tmp/GLMP/data/multiwoz/test_utterances_w_kb_w_gold.txt'

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
