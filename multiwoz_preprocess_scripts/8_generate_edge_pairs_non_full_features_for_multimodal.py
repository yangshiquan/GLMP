import numpy as np

dataset = "train"
# input_path = "../data/MULTIWOZ2.1/{}_w_kb_w_gold.txt".format(dataset)
input_path = "../data/MULTIWOZ2.1/for_debug.txt".format(dataset)
output_path = "../data/MULTIWOZ2.1/for_debug_edge_paris.txt".format(dataset)

fd = open(output_path, 'w')

n_sample = 0
kb_cnt = 0

with open(input_path) as f:
    task_type = 'multiwoz'
    start = True
    for line in f:
        line = line.strip()
        if line:
            if start:
                fd.write('#' + str(n_sample) + '\n')
                start = False
            if line.startswith("#"):
                fd.write('#' + str(n_sample) + '\n')
                line = line.replace('#', '')
                task_type = line
                continue
            nid, line = line.split(' ', 1)
            if int(nid) <= 30 and 'image' not in line:
                kb_cnt += 1
                line_list = line.split(' ')
                if task_type == 'navigate':
                    if len(line_list) == 5:
                        continue
                    elif len(line_list) == 3:
                        fd.write('[{}],[{} {}]'.format(line_list[0], line_list[1], line_list[2]) + '\n')
                        fd.write('[{} {}],[{}]'.format(line_list[1], line_list[2], line_list[0]) + '\n')
                    else:
                        continue
                elif task_type == 'schedule':
                    fd.write('[{}],[{} {}]'.format(line_list[0], line_list[1], line_list[2]) + '\n')
                    fd.write('[{} {}],[{}]'.format(line_list[1], line_list[2], line_list[0]) + '\n')
                elif task_type == 'weather':
                    if len(line_list) == 3:
                        fd.write('[{}],[{} {}]'.format(line_list[0], line_list[1], line_list[2]) + '\n')
                        fd.write('[{} {}],[{}]'.format(line_list[1], line_list[2], line_list[0]) + '\n')
                    elif len(line_list) == 4:
                        fd.write('[{}],[{} {} {}]'.format(line_list[0], line_list[1], line_list[2], line_list[3]) + '\n')
                        fd.write('[{} {} {}],[{}]'.format(line_list[1], line_list[2], line_list[3], line_list[0]) + '\n')
                    else:
                        continue
                elif task_type in ('restaurant', 'hotel', 'attraction', 'train', 'hospital', 'multiwoz'):
                    if len(line_list) == 1:
                        continue
                    elif len(line_list) == 3:
                        fd.write('[{}],[{} {}]'.format(line_list[0], line_list[1], line_list[2]) + '\n')
                        fd.write('[{} {}],[{}]'.format(line_list[1], line_list[2], line_list[0]) + '\n')
                    else:
                        continue
        else:
            if kb_cnt == 0:
                fd.write('[]' + '\n')
            kb_cnt = 0
            fd.write('\n')
            n_sample += 1
            start = True

print('success.')