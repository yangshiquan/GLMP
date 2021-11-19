import json
import ast

from utils.utils_general_kvr import *


def read_langs_multiwoz(file_name, lang, task, max_line=None):
    print(("Reading lines from {}".format(file_name)))
    data, context_arr, conv_arr, kb_arr, conv_arr_plain, kb_arr_plain = [], [], [], [], [], []
    max_resp_len = 0

    with open('data/multiwoz/multiwoz_entities.json') as f:
        global_entity = json.load(f)

    # dialogue_id_path = '/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/{}/{}_dialogue_ids.txt'.format(task, task)
    # intents_states_path = '/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/{}/{}_intents_states.json'.format(task, task)
    dialogue_id_path = '/home/yimeng/shiquan/debiasing-glmp/GLMP/data/MultiWOZ_2.2/{}/{}_dialogue_ids.txt'.format(task, task)
    intents_states_path = '/home/yimeng/shiquan/debiasing-glmp/GLMP/data/MultiWOZ_2.2/{}/{}_intents_states.json'.format(task, task)
    # dialogue_id_path = '/Users/shiquan/PycharmProjects/GLMP/data/multiwoz/{}_dialogue_ids.txt'.format(task, task)
    # intents_states_path = '/Users/shiquan/PycharmProjects/GLMP/data/multiwoz/{}_intents_states.json'.format(task, task)
    dialogue_ids = {}
    with open(dialogue_id_path, 'r') as f:
        line_cnt = 0
        for line in f:
            dialogue_ids[line_cnt] = line.strip()
            line_cnt += 1

    with open(intents_states_path, 'r') as f:
        intents_and_states = json.load(f)

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
                    gen_u = generate_memory_multiwoz(u, "$u", str(nid))
                    context_arr += gen_u
                    conv_arr += gen_u
                    conv_arr_plain.append(u)

                    annotator_id = u.rsplit(' ', 1)[-1]
                    if annotator_id not in lang.annotator2index.keys():
                        annotator_id_labels = [lang.annotator2index['NULL']] * (len(r.split())+1)
                    else:
                        annotator_id_labels = [lang.annotator2index[annotator_id]] * (len(r.split())+1)

                    dialogue_id = dialogue_ids[cnt_lin]
                    intents = intents_and_states[dialogue_id][str(turn_cnt)]['user_intents']
                    states = intents_and_states[dialogue_id][str(turn_cnt)]['dialogue_states']
                    if len(states) != 35:
                        continue
                    user_intent_labels = [1 if key in intents else 0 for key in lang.intent2index]
                    dialogue_state_labels = [[lang.state2index[key][states[key]]] for key in states]

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
                        index = [loc for loc, val in enumerate(kb_arr_plain) if (val == key and key in ent_index)]
                        if (index):
                            index = max(index)
                        else:
                            index = len(kb_arr_plain)
                        ptr_index.append(index)

                    # Get global pointer labels for words in system response, the 1 in the end is for the NULL token
                    selector_index = [1 if (word_arr in ent_index or word_arr in r.split()) else 0 for word_arr in
                                      kb_arr_plain] + [1]

                    sketch_response = generate_template_multiwoz(global_entity, r, gold_ent, kb_arr, task_type)

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
                        'sketch_response': sketch_response,
                        # 'ptr_index': ptr_index + [len(context_arr)],
                        'ptr_index': ptr_index + [len(kb_arr_plain)],
                        # 'ptr_index': ptr_index,
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
                        'ent_idx_hospital': list(set(ent_idx_hospital)),
                        'kb_arr_plain': list(kb_arr_plain + ["[NULL]"]),
                        'ent_labels': ent_labels,
                        'annotator_id_labels': annotator_id_labels,
                        'user_intent_labels': list(user_intent_labels),
                        'dialogue_state_labels': dialogue_state_labels,
                        'kb_arr_new': list(kb_arr_plain + ["[NULL]"]),
                    }
                    data.append(data_detail)

                    gen_r = generate_memory_multiwoz(r, "$s", str(nid))
                    context_arr += gen_r
                    conv_arr += gen_r
                    conv_arr_plain.append(r)
                    if max_resp_len < len(r.split()):
                        max_resp_len = len(r.split())
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
                    kb_info = generate_memory_multiwoz(r, "", str(nid))
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


def generate_template_multiwoz(global_entity, sentence, sent_ent, kb_arr, domain):
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


def generate_memory_multiwoz(sent, speaker, time):
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


def read_langs(file_name, lang, task, max_line=None):
    print(("Reading lines from {}".format(file_name)))
    data, context_arr, conv_arr, kb_arr, conv_arr_plain, kb_arr_plain = [], [], [], [], [], []
    max_resp_len = 0

    with open('data/KVR/kvret_entities.json') as f:
        global_entity = json.load(f)

    # dialogue_id_path = '/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/{}/{}_dialogue_ids.txt'.format(task, task)
    # intents_states_path = '/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/{}/{}_intents_states.json'.format(task, task)
    # dialogue_id_path = '/home/yimeng/shiquan/GLMP/data/sgd/{}/{}_dialogue_ids.txt'.format(task, task)
    # intents_states_path = '/home/yimeng/shiquan/GLMP/data/sgd/{}/{}_intents_states.json'.format(task, task)
    # dialogue_id_path = '/Users/shiquan/PycharmProjects/GLMP/data/sgd/{}_dialogue_ids.txt'.format(task, task)
    # intents_states_path = '/Users/shiquan/PycharmProjects/GLMP/data/sgd/{}_intents_states.json'.format(task, task)
    # dialogue_ids = {}
    # with open(dialogue_id_path, 'r') as f:
    #     line_cnt = 0
    #     for line in f:
    #         dialogue_ids[line_cnt] = line.strip()
    #         line_cnt += 1
    #
    # with open(intents_states_path, 'r') as f:
    #     intents_and_states = json.load(f)

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
                    gen_u = generate_memory(u, "$u", str(nid))
                    context_arr += gen_u
                    conv_arr += gen_u
                    conv_arr_plain.append(u)

                    # annotator_id = u.rsplit(' ', 1)[-1]
                    # if annotator_id not in lang.annotator2index.keys():
                    #     annotator_id_labels = [lang.annotator2index['NULL']] * (len(r.split())+1)
                    # else:
                    #     annotator_id_labels = [lang.annotator2index[annotator_id]] * (len(r.split())+1)
                    #
                    # dialogue_id = dialogue_ids[cnt_lin]
                    # intents = intents_and_states[dialogue_id][str(turn_cnt)]['user_intents']
                    # states = intents_and_states[dialogue_id][str(turn_cnt)]['dialogue_states']
                    # if len(states) != 35:
                    #     continue
                    # user_intent_labels = [1 if key in intents else 0 for key in lang.intent2index]
                    # dialogue_state_labels = [[lang.state2index[key][states[key]]] for key in states]

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

                    sketch_response = generate_template(global_entity, r, gold_ent, kb_arr, task_type)

                    # obtain gt entity labels
                    ent_labels = 0
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
                        'sketch_response': sketch_response,
                        # 'ptr_index': ptr_index + [len(context_arr)],
                        'ptr_index': ptr_index + [len(kb_arr_plain)],
                        # 'ptr_index': ptr_index,
                        'selector_index': selector_index,
                        'ent_index': ent_index,
                        'ent_idx_wet': list(set(ent_idx_wet)),
                        'ent_idx_cal': list(set(ent_idx_cal)),
                        'ent_idx_nav': list(set(ent_idx_nav)),
                        'conv_arr': list(conv_arr),
                        'conv_arr_plain': list(conv_arr_plain),
                        'kb_arr': list(kb_arr + [['$$$$'] * MEM_TOKEN_SIZE]),
                        'id': int(sample_counter),
                        'ID': int(cnt_lin),
                        'domain': task_type,
                        'kb_arr_plain': list(kb_arr_plain + ["[NULL]"]),
                        'ent_labels': ent_labels,
                        'kb_arr_new': list(kb_arr_plain + ["[NULL]"]),
                    }
                    data.append(data_detail)

                    gen_r = generate_memory(r, "$s", str(nid))
                    context_arr += gen_r
                    conv_arr += gen_r
                    conv_arr_plain.append(r)
                    if max_resp_len < len(r.split()):
                        max_resp_len = len(r.split())
                    sample_counter += 1
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
                if domain != 'weather':
                    for kb_item in kb_arr:
                        if word == kb_item[0]:
                            ent_type = kb_item[1]
                            break
                if ent_type == None:
                    for key in global_entity.keys():
                        if key!='poi':
                            global_entity[key] = [x.lower() for x in global_entity[key]]
                            if word in global_entity[key] or word.replace('_', ' ') in global_entity[key]:
                                ent_type = key
                                break
                        else:
                            poi_list = [d['poi'].lower() for d in global_entity['poi']]
                            if word in poi_list or word.replace('_', ' ') in poi_list:
                                ent_type = key
                                break
                try:
                    sketch_response.append('@'+ent_type)
                except:
                    sketch_response.append(word)
                    print(sentence)
                    print(word)
                    print(ent_type)
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
        sent_token = sent_token[::-1][:3] + ["PAD"] * (MEM_TOKEN_SIZE - 3)
        # sent_token = sent_token[::-1] + ["PAD"] * (MEM_TOKEN_SIZE - len(sent_token))
        sent_new.append(sent_token)
    return sent_new

def initialize_lang_multiwoz(lang, task):
    # path = '/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/{}/{}_intents_states.json'.format(task, task)
    path = '/home/yimeng/shiquan/debiasing-glmp/GLMP/data/MultiWOZ_2.2/{}/{}_intents_states.json'.format(task, task)
    # path = '/Users/shiquan/PycharmProjects/GLMP/data/multiwoz/{}_intents_states.json'.format(task, task)
    with open(path, 'r') as f:
        data = json.load(f)
        for id in data.keys():
            turns_data = data[id]
            for turn in turns_data.keys():
                dialogue_states = turns_data[turn]['dialogue_states']
                user_intents = turns_data[turn]['user_intents']
                for ele in user_intents:
                    lang.index_intent(ele)
                for key in dialogue_states:
                    lang.index_state_values(key, dialogue_states[key])
    if task == 'train':
        # annotator_id_info_path = '/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/MultiWOZ_2.2_Bias_ID.json'
        annotator_id_info_path = '/home/yimeng/shiquan/debiasing-glmp/GLMP/data/MultiWOZ_2.2/MultiWOZ_2.2_Bias_ID.json'
        # annotator_id_info_path = '/Users/shiquan/PycharmProjects/GLMP/data/multiwoz/MultiWOZ_2.2_Bias_ID_p=0_5.json'
        with open(annotator_id_info_path, 'r') as f:
            data = json.load(f)
            for key in data:
                annotator_id = data[key]
                lang.index_annotator(annotator_id)
        lang.index_annotator('NULL')


def initialize_lang(lang, task):
    # path = '/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/{}/{}_intents_states.json'.format(task, task)
    path = '/home/yimeng/shiquan/debiasing-glmp/GLMP/data/MultiWOZ_2.2/{}/{}_intents_states.json'.format(task, task)
    # path = '/Users/shiquan/PycharmProjects/GLMP/data/sgd/{}_intents_states.json'.format(task, task)
    with open(path, 'r') as f:
        data = json.load(f)
        for id in data.keys():
            turns_data = data[id]
            for turn in turns_data.keys():
                dialogue_states = turns_data[turn]['dialogue_states']
                user_intents = turns_data[turn]['user_intents']
                for ele in user_intents:
                    lang.index_intent(ele)
                for key in dialogue_states:
                    lang.index_state_values(key, dialogue_states[key])
    if task == 'train':
        # annotator_id_info_path = '/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/MultiWOZ_2.2_Bias_ID.json'
        annotator_id_info_path = '/home/yimeng/shiquan/debiasing-glmp/GLMP/data/MultiWOZ_2.2/MultiWOZ_2.2_Bias_ID.json'
        # annotator_id_info_path = '/Users/shiquan/PycharmProjects/GLMP/data/multiwoz/MultiWOZ_2.2_Bias_ID_p=0_5.json'
        with open(annotator_id_info_path, 'r') as f:
            data = json.load(f)
            for key in data:
                annotator_id = data[key]
                lang.index_annotator(annotator_id)
        lang.index_annotator('NULL')


def prepare_data_seq(task, batch_size=100):
    # file_train = '/Users/shiquan/PycharmProjects/GLMP/data/multiwoz/train_utterances_w_kb_w_gold_w_bias_p=0_5_sm.txt'
    # file_dev = '/Users/shiquan/PycharmProjects/GLMP/data/multiwoz/dev_utterances_w_kb_w_gold_w_bias_p=0_5_sm.txt'
    # file_test = '/Users/shiquan/PycharmProjects/GLMP/data/multiwoz/test_utterances_w_kb_w_gold_w_bias_p=0_5_sm.txt'
    # file_train = '/home/yimeng/shiquan/GLMP/data/MultiWOZ_2.2/train/aflite_filtered_samples.txt'
    file_train = '/home/yimeng/shiquan/GLMP/data/MultiWOZ_2.2/train/train_utterances_w_kb_w_gold.txt'
    file_dev = '/home/yimeng/shiquan/GLMP/data/MultiWOZ_2.2/dev/dev_utterances_w_kb_w_gold.txt'
    file_test = '/home/yimeng/shiquan/GLMP/data/MultiWOZ_2.2/test/test_utterances_w_kb_w_gold.txt'

    lang = Lang()
    for dataset in ('train', 'dev', 'test'):
        initialize_lang_multiwoz(lang, dataset)

    pair_train, train_max_len = read_langs_multiwoz(file_train, lang, 'train', max_line=None)
    train = get_seq(pair_train, lang, batch_size, True)

    lang.re_initialize()

    # file_train = '/Users/shiquan/PycharmProjects/GLMP/data/sgd/train_utterances_w_kb_w_gold.txt'
    # file_dev = '/Users/shiquan/PycharmProjects/GLMP/data/sgd/dev_utterances_w_kb_w_gold.txt'
    # file_test = '/Users/shiquan/PycharmProjects/GLMP/data/sgd/test_utterances_w_kb_w_gold.txt'
    file_train = '/home/yimeng/shiquan/GLMP/data/KVR/train_modified.txt'
    file_dev = '/home/yimeng/shiquan/GLMP/data/KVR/dev_modified.txt'
    file_test = '/home/yimeng/shiquan/GLMP/data/KVR/test_modified.txt'

    # lang = Lang()
    for dataset in ('train', 'dev', 'test'):
        initialize_lang(lang, dataset)

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
    # print(pair)
    d = get_seq(pair, lang, batch_size, False)
    return d