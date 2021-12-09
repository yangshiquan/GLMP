from utils.config import *
from models.GPT2 import *
import json


directory = args['path'].split("/")
task = directory[2].split('HDD')[0]
HDD = directory[2].split('HDD')[1].split('BSZ')[0]
L = directory[2].split('L')[1].split('lr')[0]
decoder = directory[1]
BSZ = int(directory[2].split('BSZ')[1].split('DR')[0])
DS = 'multiwoz'

if DS=='kvr':
    from utils.utils_Ent_kvr import *
elif DS=='multiwoz':
    from utils.utils_Ent_multiwoz_new_for_GPT2_finetuning import *
elif DS=='sgd':
    from utils.utils_Ent_sgd_new import *
else:
    print("You need to provide the --dataset information")

train, dev, test, testOOV, lang, max_resp_len = prepare_data_seq(task, batch_size=BSZ)

model = GPT2(
	int(HDD),
	lang,
	max_resp_len,
	args['path'],
	"",
    train.dataset.tokenizer,
	lr=0.0,
	n_layers=int(L),
	dropout=0.0)

print("STARTING GENERATING RESPONSES:")
# Set to not-training mode to disable dropout
model.train(False)

# Add evaluation and metrics calculation logic here.
total, null_total, all = 0, 0, 0
total_correct, null_total_correct, all_correct = 0, 0, 0
pbar = tqdm(enumerate(test), total=len(test))

hyp, ref = [], []
context_arr = []
for j, data_test in pbar:
    _, prob_logits = model.encoder(data_test['gpt_input'], data_test['input_arr_lengths'])
    labels = data_test['response']
    sample_ids = data_test['sample_id']
    turn_cnts = data_test['turn_cnt']
    predictions = torch.argmax(prob_logits, dim=-1)
    decoded_sentences = [" ".join(train.dataset.tokenizer.convert_ids_to_tokens(elm)) for elm in predictions.tolist()]
    # golden_sentences = [" ".join(self.tokenizer.convert_ids_to_tokens(elm)) for elm in labels.tolist()]
    for idx, sent in enumerate(decoded_sentences):
        sent_new = ' '.join([sent.split('<|endofresponse|>')[0], "<|endofresponse|>"])
        if '<|response|>' in sent_new:
            tmp = sent_new.split('<|response|>')[-1]
            tmp = tmp.strip(' ,.')
            tmp = tmp.replace('<|endofresponse|>', '')
            tmp = tmp.replace('<|endoftext|>', '')
            # tokens = train.dataset.tokenizer.encode(tmp)
            # new_tokens = []
            # for tok in tokens:
            #     if tok in train.dataset.tokenizer.encode(train.dataset.tokenizer._eos_token):
            #         continue
            #     new_tokens.append(tok)
            # response = train.dataset.tokenizer.decode(new_tokens).strip(' ,.')
            response = tmp
        else:
            response = ''
        sample_id = sample_ids[idx]
        turn_cnt = turn_cnts[idx]
        hyp.append((sample_id, turn_cnt, response))
    for idx, sent in enumerate(labels):
        sample_id = sample_ids[idx]
        turn_cnt = turn_cnts[idx]
        tmp = sent.strip()
        ref.append((sample_id, turn_cnt, tmp))

    gpt_input = data_test['gpt_input'].tolist()
    for idx, context in enumerate(gpt_input):
        context_text = ' '.join(train.dataset.tokenizer.convert_ids_to_tokens(context))
        tmp = context_text.split('<|context|>')[1].split('<|endofcontext|>')[0]
        tmp = tmp.replace('<|endoftext|>', '')
        sample_id = sample_ids[idx]
        turn_cnt = turn_cnts[idx]
        context_arr.append((sample_id, turn_cnt, tmp))

dict = {}
assert len(context_arr) == len(ref)
assert len(context_arr) == len(hyp)
for idx, elm in enumerate(context_arr):
    sample_id, turn_cnt, sentence = elm
    if sample_id not in dict:
        dict[sample_id] = {}
    if turn_cnt not in dict[sample_id]:
        dict[sample_id][turn_cnt] = {}
    dict[sample_id][turn_cnt] = {
        'context_arr': sentence,
        'gold_response': ref[idx][2],
        'generated_response': hyp[idx][2]
    }

# file_path = "/Users/shiquan/PycharmProjects/GLMP/outputs/test_generated_responses.json"
file_path = "/home/shiquan/Projects/tmp/GLMP/outputs/test_generated_responses.json"
with open(file_path, "w") as f:
    json_str = json.dumps(dict, indent=4)
    f.write(json_str)

# Set back to training mode
model.train(True)

hyp_t = [elm[2] for elm in hyp]
ref_t = [elm[2] for elm in ref]
bleu_score = moses_multi_bleu(np.array(hyp_t), np.array(ref_t), lowercase=True)
print("BLEU SCORE:\t" + str(bleu_score))
