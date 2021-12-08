from utils.config import *
from models.GPT2 import *

'''
Command:

python myTest.py -ds= -path= 

'''

directory = args['path'].split("/")
task = directory[2].split('HDD')[0]
HDD = directory[2].split('HDD')[1].split('BSZ')[0]
L = directory[2].split('L')[1].split('lr')[0]
decoder = directory[1]
BSZ = int(directory[2].split('BSZ')[1].split('DR')[0])
DS = 'multiwoz'

if DS=='kvr':
    from utils.utils_Ent_kvr import *
elif DS=='babi':
    from utils.utils_Ent_babi import *
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

acc_test = model.evaluate(test, 1e7)
if testOOV!=[]:
	acc_oov_test = model.evaluate(testOOV, 1e7)