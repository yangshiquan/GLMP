from utils.config import *
from tensorflow_models.GLMP import *
import pdb

'''
Command:

python myTest.py -ds= -path= 

'''

directory = args['path'].split("/")
task = directory[2].split('HDD')[0]
HDD = directory[2].split('HDD')[1].split('BSZ')[0]
L = directory[2].split('L')[1].split('lr')[0].split("-")[0]
decoder = directory[1].split('-')[0]
BSZ =  int(directory[2].split('BSZ')[1].split('DR')[0])
DS = 'kvr' if 'kvr' in directory[1].split('-')[1].lower() else 'babi'

early_stop = args['earlyStop']
if DS=='kvr':
    from utils.utils_tensorflow_Ent_kvr import *
elif DS=='babi':
    from utils.utils_Ent_babi import *
else:
    print("You need to provide the --dataset information")

train, dev, test, testOOV, lang, max_resp_len, train_length, dev_length, test_length = prepare_data_seq(task, batch_size=BSZ)

model = GLMP(int(HDD),
			 lang,
			 max_resp_len,
			 args['path'],
			 "",
			 lr=0.0,
			 n_layers=int(L),
			 dropout=0.0)

# len = int(test_length / int(BSZ))
test_length_ = compute_dataset_length(test_length, int(BSZ))
acc_test = model.evaluate(test, test_length_, 1e7)
if testOOV!=[]:
    acc_oov_test = model.evaluate(testOOV, 1e7)
