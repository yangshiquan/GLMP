import numpy as np
import json
import pdb

edge_dict = {}

with open('../data/KVR/train_edge_paris_full_features.txt') as f:
    for line in f:
        line = line.strip()
        if line:
            if '#' in line:
                line = line.replace('#', '')
                n_sample = line
                edge_dict[n_sample] = {}
                continue
            if line != '[]':
                head, tail = line.split(',')
                if head not in edge_dict[n_sample]:
                    edge_dict[n_sample][head] = []
                    edge_dict[n_sample][head].append(tail)
                else:
                    edge_dict[n_sample][head].append(tail)
            else:
                continue
        else:
            continue
print('success.')

fd = open('../data/KVR/train_transformed_full_features.txt', 'w')

node_list = []
n_sample = 0

with open('../data/KVR/train.txt') as f:
    for line in f:
        line = line.strip()
        if line:
            if '#' in line:
                fd.write(line + '\n')
                line = line.replace('#', '')
                task_type = line
                continue
            nid, line = line.split(' ', 1)
            if nid == '0':  # KB
                line_list = line.split(' ')
                if len(line_list) < 3:
                    continue
                if line not in node_list:
                    neighbors = edge_dict[str(n_sample)]['[{}]'.format(line)]
                    fd.write('0|[{}]|{}'.format(line, neighbors) + '\n')
                    node_list.append(line)
            else:
                fd.write('{} {}'.format(nid, line) + '\n')
        else:
            fd.write('\n')
            node_list = []
            n_sample += 1
    print('success.')