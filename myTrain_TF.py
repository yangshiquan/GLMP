from tqdm import tqdm
from utils.config import *
from tensorflow_models.GLMP import *


early_stop = args['earlyStop']
if args['dataset'] == 'kvr':
    from utils.utils_tensorflow_Ent_kvr import *
    early_stop = 'BLEU'
elif args['dataset'] == 'babi':
    from utils.utils_Ent_babi import *
    early_stop = None
    if args['task'] not in ['1', '2', '3', '4', '5']:
        print("[ERROR] You need to provide the correct --task information.")
        exit(1)
else:
    print("[Error] You need to provide the dataset information.")

print("Is there a GPU available: ", tf.test.is_gpu_available())


# ===============================
# Configure models and load data
# ===============================
avg_best, cnt, acc = 0.0, 0, 0.0
train, dev, test, testOOV, lang, max_response_len, train_length, dev_length, test_length, train_max_len, dev_max_len, test_max_len = prepare_data_seq(args['task'],
                                                                      batch_size=int(args['batch']))

# ===============================
# Build model
# ===============================
model = GLMP(int(args['hidden']),
             lang,
             max_response_len,
             args['path'],
             args['task'],
             lr=float(args['learn']),
             n_layers=int(args['layer']),
             dropout=float(args['drop']))

# ================================
# Training
# ================================
for epoch in range(200):
    print("Epoch:{}".format(epoch))
    # pdb.set_trace()
    pbar = tqdm(enumerate(train.take(-1)), total=(int(train_length/int(args['batch']))))
    for i, data in pbar:
        tf.config.experimental_run_functions_eagerly(True)
        model.train_batch(data, train_max_len, int(args['clip']), reset=(i==0))
        tf.config.experimental_run_functions_eagerly(False)
        pbar.set_description(model.print_loss())
    if ((epoch+1) % int(args['evalp']) == 0):
        len = int(dev_length / (int(args['batch'])))
        acc = model.evaluate(dev, dev_max_len, len, avg_best, early_stop)

        if (acc >= avg_best):
            avg_best = acc
            cnt = 0
        else:
            cnt += 1

        if (cnt == 8 or (acc == 1.0 and early_stop == None)):
            print("Run out of patient, early stop...")
            break