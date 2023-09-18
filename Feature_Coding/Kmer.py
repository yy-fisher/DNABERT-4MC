
def seq2kmer(seq, k):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space

    """
    kmer = [seq[x:x + k] for x in range(len(seq) + 1 - k)]
    kmers = " ".join(kmer)
    return kmers


def make_kmer_list(k, alphabet):
    """
    Generate kmer list.
    """
    if k < 0:
        print("Error, k must be an inter and larger than 0.")

    kmers = []
    for i in range(1, k + 1):
        if len(kmers) == 0:
            kmers = list(alphabet)
        else:
            new_kmers = []
            #             print('1')
            for kmer in kmers:
                for c in alphabet:
                    new_kmers.append(kmer + c)
            kmers = new_kmers
    #     print(kmers)
    return_kmers = {}
    for i, k in enumerate(kmers):
        #         print(i)
        return_kmers[k] = i

    return return_kmers