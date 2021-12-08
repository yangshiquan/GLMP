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
    predictions = torch.argmax(prob_logits, dim=-1)
    decoded_sentences = [" ".join(train.dataset.tokenizer.convert_ids_to_tokens(elm)) for elm in predictions.tolist()]
    # golden_sentences = [" ".join(self.tokenizer.convert_ids_to_tokens(elm)) for elm in labels.tolist()]
    for sent in decoded_sentences:
        sent_new = ' '.join([sent.split('<|endofresponse|>')[0], "<|endofresponse|>"])
        if '<|response|>' in sent_new:
            tmp = sent_new.split('<|response|>')[-1]
            tmp = tmp.strip(' ,.')
            tmp = tmp.replace('<|endofresponse>|', '')
            tmp = tmp.replace('<|endoftext|>', '')
            tokens = train.dataset.tokenizer.encode(tmp)
            new_tokens = []
            for tok in tokens:
                if tok in train.dataset.tokenizer.encode(train.tokenizer._eos_token):
                    continue
                new_tokens.append(tok)
            response = train.dataset.tokenizer.decode(new_tokens).strip(' ,.')
        else:
            response = ''
        hyp.append(response)
    for sent in labels:
        tmp = sent.strip()
        ref.append(tmp)

    gpt_input = data_test['gpt_input'].tolist()
    for idx, context in enumerate(gpt_input):
        context_text = train.dataset.tokenizer.convert_ids_to_tokens(context)
        tmp = context_text.split('<|context|>')[1].split('<|endofcontext|>')[0]
        tmp = tmp.replace('<|endoftext|>', '')
        context_arr.append(tmp)

dict = {}
assert len(context_arr) == len(ref)
assert len(context_arr) == len(hyp)
for idx, elm in enumerate(context_arr):
    dict[idx] = {
        'context_arr': elm,
        'gold_response': ref[idx],
        'generated_response': hyp[idx]
    }

file_path = "/Users/shiquan/PycharmProjects/GLMP/outputs/test_generated_responses.json"
with open(file_path, "w") as f:
    json_str = json.dumps(dict, indent=4)
    f.write(json_str)

# Set back to training mode
model.train(True)

bleu_score = moses_multi_bleu(np.array(hyp), np.array(ref), lowercase=True)
print("BLEU SCORE:\t" + str(bleu_score))
