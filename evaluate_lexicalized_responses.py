from utils.measures import moses_multi_bleu
import numpy as np


DS = 'multiwoz'

if DS=='kvr':
    from utils.utils_Ent_kvr import *
elif DS=='multiwoz':
    from utils.utils_Ent_multiwoz_new_for_evaluation_metrics_computation import *
elif DS=='sgd':
    from utils.utils_Ent_sgd_new import *
else:
    print("You need to provide the --dataset information")

train, dev, test, testOOV, lang, max_resp_len = prepare_data_seq('', batch_size=4)


def compute_prf(gold, pred, global_entity_list, kb_plain):
    local_kb_word = [k[0] for k in kb_plain]
    TP, FP, FN = 0, 0, 0
    if len(gold) != 0:
        count = 1
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1
        for p in set(pred):
            if p in global_entity_list or p in local_kb_word:
                if p not in gold:
                    FP += 1
        precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
        recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
        F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
    else:
        precision, recall, F1, count = 0, 0, 0, 0
    return F1, count


print("STARTING EVALUATION")


ref, hyp = [], []
acc, total = 0, 0
dialog_acc_dict = {}
F1_pred, F1_cal_pred, F1_nav_pred, F1_wet_pred, F1_restaurant_pred, F1_hotel_pred, F1_attraction_pred, F1_train_pred, F1_travel_pred, F1_events_pred, F1_weather_pred, F1_others_pred = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
F1_count, F1_cal_count, F1_nav_count, F1_wet_count, F1_restaurant_count, F1_hotel_count, F1_attraction_count, F1_train_count, F1_travel_count, F1_events_count, F1_weather_count, F1_others_count = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
pbar = tqdm(enumerate(test), total=len(test))
new_precision, new_recall, new_f1_score = 0, 0, 0
global_entity_list = []


if DS == 'kvr':
    with open('data/KVR/kvret_entities.json') as f:
        global_entity = json.load(f)
        global_entity_list = []
        for key in global_entity.keys():
            if key != 'poi':
                global_entity_list += [item.lower().replace(' ', '_') for item in global_entity[key]]
            else:
                for item in global_entity['poi']:
                    global_entity_list += [item[k].lower().replace(' ', '_') for k in item.keys()]
        global_entity_list = list(set(global_entity_list))


if DS == 'multiwoz':
    with open('data/multiwoz/multiwoz_entities.json') as f:
        global_entity = json.load(f)
        global_entity_list = []
        for key in global_entity.keys():
            global_entity_list += [item.lower().replace(' ', '_') for item in global_entity[key]]
        global_entity_list = list(set(global_entity_list))


# lex_resp_file = "/Users/shiquan/PycharmProjects/GLMP/outputs/test_generated_responses_lex.json"
lex_resp_file = "/home/shiquan/Projects/tmp/GLMP/outputs/test_generated_responses_lex.json"
with open(lex_resp_file, "r") as f:
    lex_responses = json.load(f)


for j, data_test in pbar:
    for bi in range(len(data_test['sample_id'])):
        sample_id = data_test['sample_id'][bi]
        turn_id = data_test['turn_id'][bi]
        gold_sent = lex_responses[str(sample_id)][str(turn_id)]['gold_response']
        pred_sent = lex_responses[str(sample_id)][str(turn_id)]['lexicalized_response']
        ref.append(gold_sent)
        hyp.append(pred_sent)

        if DS == 'kvr':
            # compute F1 SCORE
            single_f1, count = compute_prf(data_test['ent_index'][bi], pred_sent.split(), global_entity_list,
                                                data_test['kb_arr_plain'][bi])
            F1_pred += single_f1
            F1_count += count
            single_f1, count = compute_prf(data_test['ent_idx_cal'][bi], pred_sent.split(), global_entity_list,
                                                data_test['kb_arr_plain'][bi])
            F1_cal_pred += single_f1
            F1_cal_count += count
            single_f1, count = compute_prf(data_test['ent_idx_nav'][bi], pred_sent.split(), global_entity_list,
                                                data_test['kb_arr_plain'][bi])
            F1_nav_pred += single_f1
            F1_nav_count += count
            single_f1, count = compute_prf(data_test['ent_idx_wet'][bi], pred_sent.split(), global_entity_list,
                                                data_test['kb_arr_plain'][bi])
            F1_wet_pred += single_f1
            F1_wet_count += count
        elif DS == 'multiwoz':
            # compute F1 SCORE
            single_f1, count = compute_prf(data_test['ent_index'][bi], pred_sent.split(),
                                                global_entity_list, data_test['kb_arr_plain'][bi])  # data[14]: ent_index, data[9]: kb_arr_plain.
            F1_pred += single_f1
            F1_count += count
            single_f1, count = compute_prf(data_test['ent_idx_restaurant'][bi], pred_sent.split(),
                                                global_entity_list, data_test['kb_arr_plain'][bi])  # data[28]: ent_idx_restaurant, data[9]: kb_arr_plain.
            F1_restaurant_pred += single_f1
            F1_restaurant_count += count
            single_f1, count = compute_prf(data_test['ent_idx_hotel'][bi], pred_sent.split(),
                                                global_entity_list, data_test['kb_arr_plain'][bi])  # data[29]: ent_idx_hotel, data[9]: kb_arr_plain.
            F1_hotel_pred += single_f1
            F1_hotel_count += count
            single_f1, count = compute_prf(data_test['ent_idx_attraction'][bi], pred_sent.split(),
                                                global_entity_list, data_test['kb_arr_plain'][bi])  # data[30]: ent_idx_attraction, data[9]: kb_arr_plain.
            F1_attraction_pred += single_f1
            F1_attraction_count += count
            single_f1, count = compute_prf(data_test['ent_idx_train'][bi], pred_sent.split(),
                                                global_entity_list, data_test['kb_arr_plain'][bi])  # data[31]: ent_idx_train, data[9]: kb_arr_plain.
            F1_train_pred += single_f1
            F1_train_count += count


bleu_score = moses_multi_bleu(np.array(hyp), np.array(ref), lowercase=True)


if DS == 'kvr':
    F1_score = F1_pred / float(F1_count)
    cal_f1 = 0.0 if F1_travel_count == 0 else (F1_cal_pred / float(F1_cal_count))
    nav_f1 = 0.0 if F1_nav_count == 0 else (F1_nav_pred / float(F1_nav_count))
    wet_f1 = 0.0 if F1_events_count == 0 else (F1_wet_pred / float(F1_wet_count))
    print("F1 SCORE:\t{:.4f}".format(F1_pred / float(F1_count)))
    print("CAL F1:\t{:.4f}".format(cal_f1))
    print("NAV F1:\t{:.4f}".format(nav_f1))
    print("WET F1:\t{:.4f}".format(wet_f1))
    print("BLEU SCORE:\t" + str(bleu_score))
elif DS == 'multiwoz':
    F1_score = F1_pred / float(F1_count)
    rest_f1 = 0.0 if F1_restaurant_count == 0 else (F1_restaurant_pred / float(F1_restaurant_count))
    hotel_f1 = 0.0 if F1_hotel_count == 0 else (F1_hotel_pred / float(F1_hotel_count))
    attraction_f1 = 0.0 if F1_attraction_count == 0 else (F1_attraction_pred / float(F1_attraction_count))
    train_f1 = 0.0 if F1_train_count == 0 else (F1_train_pred / float(F1_train_count))
    print("F1 SCORE:\t{:.4f}".format(F1_pred / float(F1_count)))
    print("Restaurant F1:\t{:.4f}".format(rest_f1))
    print("Hotel F1:\t{:.4f}".format(hotel_f1))
    print("Attraction F1:\t{:.4f}".format(attraction_f1))
    print("Train F1:\t{:.4f}".format(train_f1))
    print("BLEU SCORE:\t" + str(bleu_score))
