from utils.config import *
from models.EntityPrediction import *


directory = args['path'].split("/")
task = directory[2].split('HDD')[0]
HDD = directory[2].split('HDD')[1].split('BSZ')[0]
L = directory[2].split('L')[1].split('lr')[0]
decoder = directory[1]
BSZ = int(directory[2].split('BSZ')[1].split('DR')[0])
DS = 'sgd'
args['dataset'] = DS

if DS=='kvr':
    from utils.utils_Ent_kvr_new_for_entity_prediction_at_testing_stage import *
elif DS=='multiwoz':
    from utils.utils_Ent_multiwoz_new_for_entity_prediction_at_testing_stage import *
elif DS=='sgd':
    from utils.utils_Ent_sgd_new_for_entity_prediction_at_testing_stage import *
else:
    print("You need to provide the --dataset information")

train, dev, test, testOOV, lang, max_resp_len = prepare_data_seq(task, batch_size=BSZ)

model = EntityPrediction(
	int(HDD),
	lang,
	max_resp_len,
	args['path'],
	"",
	lr=0.0,
	n_layers=int(L),
	dropout=0.0)

print("STARTING PREDICTING ENTITIES:")
# Set to not-training mode to disable dropout
model.encoder.train(False)

# Add evaluation and metrics calculation logic here.
total, null_total, all = 0, 0, 0
total_correct, null_total_correct, all_correct = 0, 0, 0
pbar = tqdm(enumerate(test), total=len(test))
dict = {}
for j, data_test in pbar:
    prob_logits = model.encoder(data_test['input'], data_test['input_arr_lengths'])
    labels = torch.tensor(data_test['target'], dtype=int)
    sample_ids = data_test['sample_id']
    turn_cnts = data_test['turn_cnt']
    timesteps = data_test['timestep']
    prob_logits[:, lang.ent2index['NULL']] = -1000000.0
    predictions = torch.argmax(prob_logits, dim=-1)
    decoded_ents = [lang.index2ent[elm] for elm in predictions.tolist()]
    golden_ents = [lang.index2ent[elm] for elm in labels.tolist()]
    for idx, elm in enumerate(decoded_ents):
        sample_id = sample_ids[idx]
        turn_cnt = turn_cnts[idx]
        timestep = timesteps[idx]
        if sample_id not in dict:
            dict[sample_id] = {}
        if turn_cnt not in dict[sample_id]:
            dict[sample_id][turn_cnt] = {}
        if timestep not in dict[sample_id][turn_cnt]:
            dict[sample_id][turn_cnt][timestep] = {
                'predicted_entity': elm
            }

# file_path = "/Users/shiquan/PycharmProjects/GLMP/outputs/test_predicted_entities.json"
file_path = "/home/shiquan/Projects/tmp/GLMP/outputs/test_predicted_entities_{}.json".format(DS)
with open(file_path, "w") as f:
    json_str = json.dumps(dict, indent=4)
    f.write(json_str)


# Set back to training mode
model.encoder.train(True)

