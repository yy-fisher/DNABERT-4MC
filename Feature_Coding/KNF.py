
def accumulated_nucle_frequency(seq):
    mapping = []
    A = 0
    C = 0
    T = 0
    G = 0
    for i in range(len(seq)):
        if seq[i] == 'A':
            A += 1
            mapping.append(A / (i + 1))
        elif seq[i] == 'C':
            C += 1
            mapping.append(C / (i + 1))
        elif seq[i] == 'T' or seq[i] == 'U':
            T += 1
            mapping.append(T / (i + 1))
        else:
            G += 1
            mapping.append(G / (i + 1))
    #     print(mapping,"W")
    padding = (41 - len(mapping))
    mapping = np.pad(mapping, (0, padding), 'constant')
    #     print(type(mapping),"q")
    #     file_record(name_seq, mapping, label)
    return mapping
