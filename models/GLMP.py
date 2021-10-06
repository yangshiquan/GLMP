import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch import optim
import torch.nn.functional as F
import random
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import os
import json

from utils.measures import wer, moses_multi_bleu
from utils.masked_cross_entropy import *
from utils.config import *
from models.modules import *


class GLMP(nn.Module):
    def __init__(self, hidden_size, lang, max_resp_len, path, task, lr, n_layers, dropout):
        super(GLMP, self).__init__()
        self.name = "GLMP"
        self.task = task
        self.input_size = lang.n_words
        self.output_size = lang.n_words
        self.hidden_size = hidden_size    
        self.lang = lang
        self.lr = lr
        self.n_layers = n_layers
        self.dropout = dropout
        self.max_resp_len = max_resp_len
        self.decoder_hop = n_layers
        self.softmax = nn.Softmax(dim=0)

        if path:
            if USE_CUDA:
                print("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path)+'/enc.th')
                self.extKnow = torch.load(str(path)+'/enc_kb.th')
                self.decoder = torch.load(str(path)+'/dec.th')
                self.entPred = torch.load(str(path)+'/ent_pred.th')
                self.clEntPred = torch.load(str(path)+'/cl_ent_pred.th')
            else:
                print("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path)+'/enc.th',lambda storage, loc: storage)
                self.extKnow = torch.load(str(path)+'/enc_kb.th',lambda storage, loc: storage)
                self.decoder = torch.load(str(path)+'/dec.th',lambda storage, loc: storage)
                self.entPred = torch.load(str(path) + '/ent_pred.th', lambda storage, loc: storage)
                self.clEntPred = torch.load(str(path) + '/cl_ent_pred.th', lambda storage, loc: storage)
        else:
            self.encoder = ContextRNN(lang.n_words, hidden_size, dropout)
            self.extKnow = ExternalKnowledge(lang.n_words, hidden_size, n_layers, dropout)
            self.decoder = LocalMemoryDecoder(self.encoder.embedding, lang, hidden_size, self.decoder_hop, dropout) #Generator(lang, hidden_size, dropout)
            self.entPred = EntityPredictionRNN(lang.n_words, hidden_size, dropout, self.encoder.embedding, lang.n_annotators)
            self.clEntPred = torch.load('/Users/shiquan/PycharmProjects/GLMP/save/GLMP-CL_PRETRAIN/HDD128BSZ8DR0.2L1lr0.001EPOCH-0-LOSS-tensor(0.5862, grad_fn=<NllLossBackward>)/cl_enc.th')
            for param in self.clEntPred.parameters():
                param.requires_grad = False
            # self.entPred = torch.load('/home/shiquan/Projects/debias-glmp/GLMP/save/GLMP-KVR/HDD128BSZ4DR0.2L1lr0.001PRETRAINED/ent_pred.th')

        # Initialize optimizers and criterion
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.extKnow_optimizer = optim.Adam(self.extKnow.parameters(), lr=lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr)
        self.entPred_optimizer = optim.Adam(self.decoder.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer, mode='max', factor=0.5, patience=1, min_lr=0.0001, verbose=True)
        self.criterion_bce = nn.BCELoss()
        self.reset()

        if USE_CUDA:
            self.encoder.cuda()
            self.extKnow.cuda()
            self.decoder.cuda()
            self.entPred.cuda()
            self.clEntPred.cuda()

    def print_loss(self):    
        print_loss_avg = self.loss / self.print_every
        print_loss_b = self.loss_b / self.print_every
        print_loss_v = self.loss_v / self.print_every
        print_loss_l = self.loss_l / self.print_every
        self.print_every += 1
        return 'L:{:.2f},LE:{:.2f},LB:{:.2f},LP:{:.2f}'.format(print_loss_avg, print_loss_v, print_loss_b, print_loss_l)
    
    def save_model(self, dec_type):
        name_data = "KVR/" if self.task=='' else "BABI/"
        layer_info = str(self.n_layers)
        directory = 'save/GLMP-'+args["addName"]+name_data+str(self.task)+'HDD'+str(self.hidden_size)+'BSZ'+str(args['batch'])+'DR'+str(self.dropout)+'L'+layer_info+'lr'+str(self.lr)+str(dec_type)                 
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder, directory + '/enc.th')
        torch.save(self.extKnow, directory + '/enc_kb.th')
        torch.save(self.decoder, directory + '/dec.th')
        torch.save(self.entPred, directory + '/ent_pred.th')
        torch.save(self.clEntPred, directory + '/cl_ent_pred.th')

    def reset(self):
        self.loss, self.print_every, self.loss_b, self.loss_v, self.loss_l = 0, 1, 0, 0, 0
    
    def _cuda(self, x):
        if USE_CUDA:
            return torch.Tensor(x).cuda()
        else:
            return torch.Tensor(x)

    def train_batch(self, data, clip, reset=0):
        if reset: self.reset()
        # Zero gradients of both optimizers
        self.encoder_optimizer.zero_grad()
        self.extKnow_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        self.entPred_optimizer.zero_grad()
        
        # Encode and Decode
        use_teacher_forcing = random.random() < args['teacher_forcing_ratio'] 
        max_target_length = max(data['response_lengths'])
        all_decoder_outputs_vocab, all_decoder_outputs_ptr, _, _, global_pointer, all_decoder_outputs_ptr_biased = self.encode_and_decode(data, max_target_length, use_teacher_forcing, False)

        # Loss calculation and backpropagation
        loss_v = masked_cross_entropy(
            all_decoder_outputs_vocab.transpose(0, 1).contiguous(),
            data['sketch_response'].contiguous(),
            data['response_lengths'])
        loss_l = masked_cross_entropy(
            all_decoder_outputs_ptr.transpose(0, 1).contiguous(),
            data['ptr_index'].contiguous(),
            data['response_lengths'])
        loss_b = masked_cross_entropy(
            all_decoder_outputs_ptr_biased.transpose(0, 1).contiguous(),
            data['ptr_index'].contiguous(),
            data['response_lengths'])
        loss = loss_l + loss_v + loss_b
        loss.backward()

        # Clip gradient norms
        ec = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
        kc = torch.nn.utils.clip_grad_norm_(self.extKnow.parameters(), clip)
        dc = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)
        pc = torch.nn.utils.clip_grad_norm_(self.entPred.parameters(), clip)

        # Update parameters with optimizers
        self.encoder_optimizer.step()
        self.extKnow_optimizer.step()
        self.decoder_optimizer.step()
        self.entPred_optimizer.step()
        self.loss += loss.item()
        self.loss_b += loss_b.item()
        self.loss_v += loss_v.item()
        self.loss_l += loss_l.item()

    def encode_and_decode(self, data, max_target_length, use_teacher_forcing, get_decoded_words):
        # Build unknown mask for memory
        if args['unk_mask'] and self.decoder.training:
            story_size = data['context_arr'].size()
            rand_mask = np.ones(story_size)
            bi_mask = np.random.binomial([np.ones((story_size[0],story_size[1]))], 1-self.dropout)[0]
            rand_mask[:,:,0] = rand_mask[:,:,0] * bi_mask
            conv_rand_mask = np.ones(data['conv_arr'].size())
            for bi in range(story_size[0]):
                start, end = data['kb_arr_lengths'][bi],  data['kb_arr_lengths'][bi] + data['conv_arr_lengths'][bi]
                conv_rand_mask[:end-start,bi,:] = rand_mask[bi,start:end,:]
            rand_mask = self._cuda(rand_mask)
            conv_rand_mask = self._cuda(conv_rand_mask)
            conv_story = data['conv_arr'] * conv_rand_mask.long()
            story = data['context_arr'] * rand_mask.long()
        else:
            story, conv_story = data['context_arr'], data['conv_arr']
        
        # Encode dialog history and KB to vectors
        dh_outputs, dh_hidden = self.encoder(conv_story, data['conv_arr_lengths'])
        global_pointer = None
        encoded_hidden = torch.cat((dh_hidden.squeeze(0), dh_hidden.squeeze(0)), dim=1)

        # Get the words that can be copy from the memory
        batch_size = len(data['context_arr_lengths'])
        self.copy_list = data['kb_arr_plain_new']
        # self.copy_list = []
        # for elm in data['context_arr_plain']:
        #     elm_temp = [ word_arr[0] for word_arr in elm ]
        #     self.copy_list.append(elm_temp)

        outputs_vocab, outputs_ptr, decoded_fine, decoded_coarse, outputs_ptr_biased = self.decoder(
            self.entPred,
            # self.debiasedKnow,
            story.size(),
            data['context_arr_lengths'],
            self.copy_list,
            encoded_hidden,
            data['sketch_response'],
            max_target_length,
            batch_size,
            use_teacher_forcing,
            get_decoded_words,
            global_pointer,
            data['conv_arr_plain'],
            data['kb_arr_plain_new'],
            data['kb_arr_new'],
            torch.Tensor(data['ent_labels']).long(),
            conv_story,
            data['conv_arr_lengths'],
            self.clEntPred
        )

        return outputs_vocab, outputs_ptr, decoded_fine, decoded_coarse, global_pointer, outputs_ptr_biased

    def evaluate(self, dev, matric_best, early_stop=None):
        print("STARTING EVALUATION")
        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.extKnow.train(False)
        self.decoder.train(False)
        self.entPred.train(False)
        
        ref, hyp = [], []
        acc, total = 0, 0
        dialog_acc_dict = {}
        F1_pred, F1_cal_pred, F1_nav_pred, F1_wet_pred, F1_restaurant_pred, F1_hotel_pred, F1_attraction_pred, F1_train_pred, F1_travel_pred, F1_events_pred, F1_weather_pred, F1_others_pred = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        F1_count, F1_cal_count, F1_nav_count, F1_wet_count, F1_restaurant_count, F1_hotel_count, F1_attraction_count, F1_train_count, F1_travel_count, F1_events_count, F1_weather_count, F1_others_count = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        pbar = tqdm(enumerate(dev),total=len(dev))
        new_precision, new_recall, new_f1_score = 0, 0, 0
        global_entity_list = []

        if args['dataset'] == 'kvr':
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

        if args['dataset'] == 'multiwoz':
            with open('data/multiwoz/multiwoz_entities.json') as f:
                global_entity = json.load(f)
                global_entity_list = []
                for key in global_entity.keys():
                    global_entity_list += [item.lower().replace(' ', '_') for item in global_entity[key]]
                global_entity_list = list(set(global_entity_list))

        if args['dataset'] == 'sgd':
            with open('data/sgd/sgd_entities.json') as f:
                global_entity = json.load(f)
                global_entity_list = []
                for key in global_entity.keys():
                    global_entity_list += [item.lower().replace(' ', '_') for item in global_entity[key]]
                global_entity_list = list(set(global_entity_list))

        for j, data_dev in pbar:
            # Encode and Decode
            max_target_length = max(data_dev['response_lengths'])
            _, _, decoded_fine, decoded_coarse, global_pointer, _ = self.encode_and_decode(data_dev, max_target_length, False, True)
            decoded_coarse = np.transpose(decoded_coarse)
            decoded_fine = np.transpose(decoded_fine)
            for bi, row in enumerate(decoded_fine):
                st = ''
                for e in row:
                    if e == 'EOS': break
                    else: st += e + ' '
                st_c = ''
                for e in decoded_coarse[bi]:
                    if e == 'EOS': break
                    else: st_c += e + ' '
                pred_sent = st.lstrip().rstrip()
                pred_sent_coarse = st_c.lstrip().rstrip()
                gold_sent = data_dev['response_plain'][bi].lstrip().rstrip()
                ref.append(gold_sent)
                hyp.append(pred_sent)
                
                if args['dataset'] == 'kvr':
                    # compute F1 SCORE
                    single_f1, count = self.compute_prf(data_dev['ent_index'][bi], pred_sent.split(), global_entity_list, data_dev['kb_arr_plain'][bi])
                    F1_pred += single_f1
                    F1_count += count
                    single_f1, count = self.compute_prf(data_dev['ent_idx_cal'][bi], pred_sent.split(), global_entity_list, data_dev['kb_arr_plain'][bi])
                    F1_cal_pred += single_f1
                    F1_cal_count += count
                    single_f1, count = self.compute_prf(data_dev['ent_idx_nav'][bi], pred_sent.split(), global_entity_list, data_dev['kb_arr_plain'][bi])
                    F1_nav_pred += single_f1
                    F1_nav_count += count
                    single_f1, count = self.compute_prf(data_dev['ent_idx_wet'][bi], pred_sent.split(), global_entity_list, data_dev['kb_arr_plain'][bi])
                    F1_wet_pred += single_f1
                    F1_wet_count += count
                elif args['dataset'] == 'multiwoz':
                    # compute F1 SCORE
                    single_f1, count = self.compute_prf(data_dev['ent_index'][bi], pred_sent.split(),
                                                        global_entity_list, data_dev['kb_arr_plain'][bi])  # data[14]: ent_index, data[9]: kb_arr_plain.
                    F1_pred += single_f1
                    F1_count += count
                    single_f1, count = self.compute_prf(data_dev['ent_idx_restaurant'][bi], pred_sent.split(),
                                                        global_entity_list, data_dev['kb_arr_plain'][bi])  # data[28]: ent_idx_restaurant, data[9]: kb_arr_plain.
                    F1_restaurant_pred += single_f1
                    F1_restaurant_count += count
                    single_f1, count = self.compute_prf(data_dev['ent_idx_hotel'][bi], pred_sent.split(),
                                                        global_entity_list, data_dev['kb_arr_plain'][bi])  # data[29]: ent_idx_hotel, data[9]: kb_arr_plain.
                    F1_hotel_pred += single_f1
                    F1_hotel_count += count
                    single_f1, count = self.compute_prf(data_dev['ent_idx_attraction'][bi], pred_sent.split(),
                                                        global_entity_list, data_dev['kb_arr_plain'][bi])  # data[30]: ent_idx_attraction, data[9]: kb_arr_plain.
                    F1_attraction_pred += single_f1
                    F1_attraction_count += count
                    single_f1, count = self.compute_prf(data_dev['ent_idx_train'][bi], pred_sent.split(),
                                                        global_entity_list, data_dev['kb_arr_plain'][bi])  # data[31]: ent_idx_train, data[9]: kb_arr_plain.
                    F1_train_pred += single_f1
                    F1_train_count += count
                elif args['dataset'] == 'sgd':
                    # compute F1 SCORE
                    single_f1, count = self.compute_prf(data_dev['ent_index'][bi], pred_sent.split(),
                                                        global_entity_list, data_dev['kb_arr_plain'][bi])  # data[14]: ent_index, data[9]: kb_arr_plain.
                    F1_pred += single_f1
                    F1_count += count
                    single_f1, count = self.compute_prf(data_dev['ent_idx_travel'][bi], pred_sent.split(),
                                                        global_entity_list, data_dev['kb_arr_plain'][bi])  # data[28]: ent_idx_restaurant, data[9]: kb_arr_plain.
                    F1_travel_pred += single_f1
                    F1_travel_count += count
                    single_f1, count = self.compute_prf(data_dev['ent_idx_hotel'][bi], pred_sent.split(),
                                                        global_entity_list, data_dev['kb_arr_plain'][bi])  # data[29]: ent_idx_hotel, data[9]: kb_arr_plain.
                    F1_hotel_pred += single_f1
                    F1_hotel_count += count
                    single_f1, count = self.compute_prf(data_dev['ent_idx_events'][bi], pred_sent.split(),
                                                        global_entity_list, data_dev['kb_arr_plain'][bi])  # data[30]: ent_idx_attraction, data[9]: kb_arr_plain.
                    F1_events_pred += single_f1
                    F1_events_count += count
                    single_f1, count = self.compute_prf(data_dev['ent_idx_weather'][bi], pred_sent.split(),
                                                        global_entity_list, data_dev['kb_arr_plain'][bi])  # data[31]: ent_idx_train, data[9]: kb_arr_plain.
                    F1_weather_pred += single_f1
                    F1_weather_count += count
                    single_f1, count = self.compute_prf(data_dev['ent_idx_others'][bi], pred_sent.split(),
                                                        global_entity_list, data_dev['kb_arr_plain'][bi])  # data[31]: ent_idx_train, data[9]: kb_arr_plain.
                    F1_others_pred += single_f1
                    F1_others_count += count
                else:
                    # compute Dialogue Accuracy Score
                    current_id = data_dev['ID'][bi]
                    if current_id not in dialog_acc_dict.keys():
                        dialog_acc_dict[current_id] = []
                    if gold_sent == pred_sent:
                        dialog_acc_dict[current_id].append(1)
                    else:
                        dialog_acc_dict[current_id].append(0)

                # compute Per-response Accuracy Score
                total += 1
                if (gold_sent == pred_sent):
                    acc += 1

                if args['genSample']:
                    self.print_examples(bi, data_dev, pred_sent, pred_sent_coarse, gold_sent)

        # Set back to training mode
        self.encoder.train(True)
        self.extKnow.train(True)
        self.decoder.train(True)
        self.entPred.train(True)

        bleu_score = moses_multi_bleu(np.array(hyp), np.array(ref), lowercase=True)
        acc_score = acc / float(total)
        print("ACC SCORE:\t"+str(acc_score))

        if args['dataset'] == 'kvr':
            F1_score = F1_pred / float(F1_count)
            cal_f1 = 0.0 if F1_travel_count == 0 else (F1_cal_pred / float(F1_cal_count))
            nav_f1 = 0.0 if F1_nav_count == 0 else (F1_nav_pred / float(F1_nav_count))
            wet_f1 = 0.0 if F1_events_count == 0 else (F1_wet_pred / float(F1_wet_count))
            print("F1 SCORE:\t{:.4f}".format(F1_pred / float(F1_count)))
            print("CAL F1:\t{:.4f}".format(cal_f1))
            print("NAV F1:\t{:.4f}".format(nav_f1))
            print("WET F1:\t{:.4f}".format(wet_f1))
            print("BLEU SCORE:\t" + str(bleu_score))
        elif args['dataset'] == 'multiwoz':
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
        elif args['dataset'] == 'sgd':
            F1_score = F1_pred / float(F1_count)
            travel_f1 = 0.0 if F1_travel_count == 0 else (F1_restaurant_pred / float(F1_travel_count))
            hotel_f1 = 0.0 if F1_hotel_count == 0 else (F1_hotel_pred / float(F1_hotel_count))
            events_f1 = 0.0 if F1_events_count == 0 else (F1_events_pred / float(F1_events_count))
            weather_f1 = 0.0 if F1_weather_count == 0 else (F1_weather_pred / float(F1_weather_count))
            others_f1 = 0.0 if F1_others_count == 0 else (F1_others_pred / float(F1_others_count))
            print("F1 SCORE:\t{:.4f}".format(F1_pred / float(F1_count)))
            print("Travel F1:\t{:.4f}".format(travel_f1))
            print("Hotel F1:\t{:.4f}".format(hotel_f1))
            print("Events F1:\t{:.4f}".format(events_f1))
            print("Weather F1:\t{:.4f}".format(weather_f1))
            print("Others F1:\t{:.4f}".format(others_f1))
            print("BLEU SCORE:\t" + str(bleu_score))
        else:
            dia_acc = 0
            for k in dialog_acc_dict.keys():
                if len(dialog_acc_dict[k])==sum(dialog_acc_dict[k]):
                    dia_acc += 1
            print("Dialog Accuracy:\t"+str(dia_acc*1.0/len(dialog_acc_dict.keys())))
        
        if (early_stop == 'BLEU'):
            if (bleu_score >= matric_best):
                self.save_model('BLEU-'+str(bleu_score))
                print("MODEL SAVED")
            return bleu_score
        elif (early_stop == 'ENTF1'):
            if (F1_score >= matric_best):
                self.save_model('ENTF1-{:.4f}'.format(F1_score))
                print("MODEL SAVED")  
            return F1_score
        else:
            if (acc_score >= matric_best):
                self.save_model('ACC-{:.4f}'.format(acc_score))
                print("MODEL SAVED")
            return acc_score

    def compute_prf(self, gold, pred, global_entity_list, kb_plain):
        local_kb_word = [k[0] for k in kb_plain]
        TP, FP, FN = 0, 0, 0
        if len(gold)!= 0:
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
            precision = TP / float(TP+FP) if (TP+FP)!=0 else 0
            recall = TP / float(TP+FN) if (TP+FN)!=0 else 0
            F1 = 2 * precision * recall / float(precision + recall) if (precision+recall)!=0 else 0
        else:
            precision, recall, F1, count = 0, 0, 0, 0
        return F1, count

    def print_examples(self, batch_idx, data, pred_sent, pred_sent_coarse, gold_sent):
        kb_len = len(data['context_arr_plain'][batch_idx])-data['conv_arr_lengths'][batch_idx]-1
        print("{}: ID{} id{} ".format(data['domain'][batch_idx], data['ID'][batch_idx], data['id'][batch_idx]))
        for i in range(kb_len): 
            kb_temp = [w for w in data['context_arr_plain'][batch_idx][i] if w!='PAD']
            kb_temp = kb_temp[::-1]
            if 'poi' not in kb_temp:
                print(kb_temp)
        flag_uttr, uttr = '$u', []
        for word_idx, word_arr in enumerate(data['context_arr_plain'][batch_idx][kb_len:]):
            if word_arr[1]==flag_uttr:
                uttr.append(word_arr[0])
            else:
                print(flag_uttr,': ', " ".join(uttr))
                flag_uttr = word_arr[1]
                uttr = [word_arr[0]]
        print('Sketch System Response : ', pred_sent_coarse)
        print('Final System Response : ', pred_sent)
        print('Gold System Response : ', gold_sent)
        print('\n')
