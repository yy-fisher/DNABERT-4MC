import re
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from Feature_Coding.Kmer import *
from Feature_Coding.NCP import *
from Feature_Coding.KNF import *
def READ_FASTA(data_path):
    """return: dict {
        name: [seqnece , target]
    }"""
    alphabet = ("N|Y|M|N|X|Z")
    removelist = []
    with open(data_path) as Data_file:

        file_str = Data_file.read()
        all_sequence = file_str.split('\n')
        len_sequence = len(all_sequence)
        #         print(len_sequence)
        sequence_dict = {}
        for i in range(0, len_sequence - 2, 2):
            if all_sequence[i] == '[]':
                del all_sequence[i]
            if re.search(alphabet, str(all_sequence[i + 1])) is not None:
                print(all_sequence[i], "REMOVE")
                removelist.append(all_sequence[i])
            if all_sequence[i].split('|')[1] == "1":
                seq_label = 1
            else:
                seq_label = 0
            sequence_dict[all_sequence[i]] = [all_sequence[i + 1], seq_label]
        for i in removelist:
            del sequence_dict[i]
        return sequence_dict


def get_data(data_path):
    src_vocab1 = make_kmer_list(1, 'ATCG')
    print(src_vocab1, "o")
    src = src_vocab1.copy()
    src_vocab3 = Eiip(src)

    #     print(src_vocab1,"o")
    src_vocab2 = make_kmer_list(2, 'ATCG')
    src_vocab4 = make_kmer_list(6, 'ATCG')
    print(src_vocab4, "w")
    print(src_vocab2)
    src_vocab_size1 = len(src_vocab1)
    src_vocab_size2 = len(src_vocab2)
    src_vocab_size4 = len(src_vocab4)
    seq_dict = READ_FASTA(data_path)
    #     print(seq_dict)
    seq_data = pd.DataFrame(seq_dict).T
    seq_data.columns = ['seq', 'label']
    seq_data = seq_data.reset_index(drop=True)

    #     x= Eiip(seq_data.seq)
    #     print(x)
    seq_data['seq_1'] = seq_data.seq.apply(lambda x: np.array([src_vocab1[x[w:w + 1]] for w in range(len(x) - 1 + 1)]))
    seq_data['seq_2'] = seq_data.seq.apply(lambda x: np.array([src_vocab2[x[w:w + 2]] for w in range(len(x) - 2 + 1)]))
    seq_data['seq_2_1'] = seq_data.seq.apply(
        lambda x: np.array([src_vocab2[x[w] + x[w + 2]] for w in range(len(x) - 3 + 1)]))
    seq_data['seq_2_2'] = seq_data.seq.apply(
        lambda x: np.array([src_vocab2[x[w] + x[w + 3]] for w in range(len(x) - 4 + 1)]))
    seq_data['seq_2_3'] = seq_data.seq.apply(
        lambda x: np.array([src_vocab2[x[w] + x[w + 4]] for w in range(len(x) - 5 + 1)]))
    seq_data['seq_2_4'] = seq_data.seq.apply(
        lambda x: np.array([src_vocab3[x[w:w + 1]] for w in range(len(x) - 1 + 1)]))
    seq_data['seq_2_5'] = seq_data.seq.apply(lambda x: accumulated_nucle_frequency(x))
    seq_data['seq_2_6'] = seq_data["seq_2_4"] * seq_data["seq_2_5"]
    seq_data['seq_2_7'] = seq_data.seq.apply(
        lambda x: np.array([src_vocab4[x[w:w + 6]] for w in range(len(x) - 6 + 1)]))
    return seq_data

dataset_path = '/DataSets/4mC_G.subterraneus.txt'
dataset = READ_FASTA(dataset_path)
kf = KFold(n_splits = 10, shuffle=True, random_state=0)
for train_index, valid_index in kf.split(dataset):
    print(len(train_index),len(valid_index))
dataset_dna ='DataSets/4mC_G.subterraneus.txt'
seq_dataset = get_data(dataset_dna)
seq_train = seq_dataset.iloc[train_index]
seq_valid = seq_dataset.iloc[valid_index]
seq_dataset_bert  = seq_dataset[['seq','label']]