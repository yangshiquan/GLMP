import json
import ast

from utils.utils_general import *


def read_langs(file_name, max_line=None):
    print(("Reading lines from {}".format(file_name)))
    data, context_arr, conv_arr, kb_arr, conv_arr_plain = [], [], [], [], []
    max_resp_len = 0

    with open('data/multiwoz/multiwoz_entities.json') as f:
        global_entity = json.load(f)

    with open(file_name) as fin:
        cnt_lin, sample_counter = 1, 1
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
                    ent_idx_restaurant, ent_idx_hotel, ent_idx_attraction, ent_idx_train, ent_idx_hospital = [], [], [], [], []
                    if task_type == "restaurant":
                        ent_idx_restaurant = gold_ent
                    elif task_type == "hotel":
                        ent_idx_hotel = gold_ent
                    elif task_type == "attraction":
                        ent_idx_attraction = gold_ent
                    elif task_type == "train":
                        ent_idx_train = gold_ent
                    elif task_type == "hospital":
                        ent_idx_hospital = gold_ent
                    ent_index = list(
                        set(ent_idx_restaurant + ent_idx_hotel + ent_idx_attraction + ent_idx_train + ent_idx_hospital))

                    # Get local pointer position for each word in system response
                    ptr_index = []
                    for key in r.split():
                        index = [loc for loc, val in enumerate(context_arr) if (val[0] == key and key in ent_index)]
                        if (index):
                            index = max(index)
                        else:
                            index = len(context_arr)
                        ptr_index.append(index)

                    # Get global pointer labels for words in system response, the 1 in the end is for the NULL token
                    selector_index = [1 if (word_arr[0] in ent_index or word_arr[0] in r.split()) else 0 for word_arr in
                                      context_arr] + [1]

                    sketch_response = generate_template(global_entity, r, gold_ent, kb_arr, task_type)

                    data_detail = {
                        'context_arr': list(context_arr + [['$$$$'] * MEM_TOKEN_SIZE]),  # $$$$ is NULL token
                        'response': r,
                        'sketch_response': sketch_response,
                        'ptr_index': ptr_index + [len(context_arr)],
                        'selector_index': selector_index,
                        'ent_index': ent_index,
                        'ent_idx_cal': list(set(ent_idx_cal)),
                        'ent_idx_nav': list(set(ent_idx_nav)),
                        'ent_idx_wet': list(set(ent_idx_wet)),
                        'conv_arr': list(conv_arr),
                        'conv_arr_plain': list(conv_arr_plain),
                        'kb_arr': list(kb_arr + [['$$$$'] * MEM_TOKEN_SIZE]),
                        'id': int(sample_counter),
                        'ID': int(cnt_lin),
                        'domain': task_type,
                        'ent_idx_restaurant': list(set(ent_idx_restaurant)),
                        'ent_idx_hotel': list(set(ent_idx_hotel)),
                        'ent_idx_attraction': list(set(ent_idx_attraction)),
                        'ent_idx_train': list(set(ent_idx_train)),
                        'ent_idx_hospital': list(set(ent_idx_hospital))}
                    data.append(data_detail)

                    gen_r = generate_memory(r, "$s", str(nid))
                    context_arr += gen_r
                    conv_arr += gen_r
                    conv_arr_plain.append(r)
                    if max_resp_len < len(r.split()):
                        max_resp_len = len(r.split())
                    sample_counter += 1
                else:
                    # deal with knowledge graph
                    r = line
                    kb_info = generate_memory(r, "", str(nid))
                    if len(kb_info[0]) > 4:
                        print(kb_info)
                        print(r)
                    context_arr = kb_info + context_arr
                    kb_arr += kb_info
            else:
                cnt_lin += 1
                context_arr, conv_arr, kb_arr, conv_arr_plain = [], [], [], []
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
    file_train = '/Users/shiquan/PycharmProjects/GLMP/data/multiwoz/train.txt'
    file_dev = '/Users/shiquan/PycharmProjects/GLMP/data/multiwoz/valid.txt'
    file_test = '/Users/shiquan/PycharmProjects/GLMP/data/multiwoz/test.txt'

    pair_train, train_max_len = read_langs(file_train, max_line=None)
    pair_dev, dev_max_len = read_langs(file_dev, max_line=None)
    pair_test, test_max_len = read_langs(file_test, max_line=None)
    max_resp_len = max(train_max_len, dev_max_len, test_max_len) + 1

    lang = Lang()

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