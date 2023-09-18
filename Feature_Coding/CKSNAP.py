def encoding(seq_dataset):
    categorylen = 1+3+1+4
    probMatr = np.zeros((len(dataset) ,41,categorylen))
    sampleNo = 0
    for sequence in seq_dataset["seq"]:
        AANo = 0
        sequencestr = ''
        for aa in sequence:
            sequencestr+=aa
            if aa == ' ':
                sequencestr =''
#         print(sequencestr)
        for j in range(len(sequencestr)):
            thisLetter = sequencestr[j]
            if(thisLetter == "A"):
                probMatr[sampleNo][AANo][0] = 1
                probMatr[sampleNo][AANo][1] = 1
                probMatr[sampleNo][AANo][2] = 1
                probMatr[sampleNo][AANo][3] = 0.126
                probMatr[sampleNo][AANo][4] = 1
                probMatr[sampleNo][AANo][5] = 0
                probMatr[sampleNo][AANo][6] = 0
                probMatr[sampleNo][AANo][7] = 0
            elif(thisLetter == "C"):
                probMatr[sampleNo][AANo][0] = 0
                probMatr[sampleNo][AANo][1] = 1
                probMatr[sampleNo][AANo][2] = 0
                probMatr[sampleNo][AANo][3] = 0.134
                probMatr[sampleNo][AANo][4] = 0
                probMatr[sampleNo][AANo][5] = 1
                probMatr[sampleNo][AANo][6] = 0
                probMatr[sampleNo][AANo][7] = 0
            elif(thisLetter == "G"):
                probMatr[sampleNo][AANo][0] = 1
                probMatr[sampleNo][AANo][1] = 0
                probMatr[sampleNo][AANo][2] = 0
                probMatr[sampleNo][AANo][3] = 0.0806
                probMatr[sampleNo][AANo][4] = 0
                probMatr[sampleNo][AANo][5] = 0
                probMatr[sampleNo][AANo][6] = 1
                probMatr[sampleNo][AANo][7] = 0
            elif(thisLetter == "T"):
                probMatr[sampleNo][AANo][0] = 0
                probMatr[sampleNo][AANo][1] = 0
                probMatr[sampleNo][AANo][2] = 1
                probMatr[sampleNo][AANo][3] = 0.1335
                probMatr[sampleNo][AANo][4] = 0
                probMatr[sampleNo][AANo][5] = 0
                probMatr[sampleNo][AANo][6] = 0
                probMatr[sampleNo][AANo][7] = 1
            else:
                probMatr[sampleNo][AANo][0] = 0
                probMatr[sampleNo][AANo][1] = 0
                probMatr[sampleNo][AANo][2] = 0
                probMatr[sampleNo][AANo][3] = 0
                probMatr[sampleNo][AANo][4] = 0
                probMatr[sampleNo][AANo][5] = 0
                probMatr[sampleNo][AANo][6] = 0
                probMatr[sampleNo][AANo][7] = 0

            probMatr[sampleNo][AANo][8] = sequencestr[0: j + 1].count(sequencestr[j]) / float(j + 1)

            AANo += 1
        sampleNo += 1
    return probMatr

aneo = encoding(seq_dataset)
aneo = aneo.reshape(len(dataset) ,-1)
for i in range(seq_dataset.shape[0]):
    seq_dataset['seq_2_3'][i] = aneo[i]