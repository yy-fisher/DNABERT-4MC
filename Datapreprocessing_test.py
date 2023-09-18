# 测试集制作
def READ_TEST_FASTA(data_path):
    """return: dict {
        name: [seqnece , target]
    }"""

    with open(data_path) as Data_file:

        file_str = Data_file.read()

        all_sequence = file_str.split('\n')
        len_sequence = len(all_sequence)
        sequence_dict = {}
        for i in range(0, len_sequence - 1, 2):
            #             print(all_sequence[i])
            if all_sequence[i][:2] == ">P":
                seq_label = 1
            else:
                seq_label = 0
            sequence_dict[all_sequence[i]] = [all_sequence[i + 1], seq_label]

        return sequence_dict