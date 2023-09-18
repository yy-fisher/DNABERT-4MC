
def Eiip(return_kmer):
    for i in return_kmer:
        if i == 'A':
            return_kmer[i] = 0.1260
        if i == 'C':
            return_kmer[i] = 0.1340
        if i == 'G':
            return_kmer[i] = 0.0806
        if i == 'T':
            return_kmer[i] = 0.1335

    return return_kmer


