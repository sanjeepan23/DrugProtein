import re
import math
import numpy as np
from collections import Counter

def AAC(fastas, **kw):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    #AA = 'ARNDCQEGHILKMFPSTWYV'
    encodings = []
    header = []
    for i in AA:
        header.append(i)
    #encodings.append(header)

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        count = Counter(sequence)
        for key in count:
            count[key] = count[key]/len(sequence)
        code = []
        for aa in AA:
            code.append(count[aa])
        encodings.append(code)
    return np.array(encodings, dtype=float), header


def Rvalue(aa1, aa2, AADict, Matrix):
    return sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2 for i in range(len(Matrix))]) / len(Matrix)


def PAAC(fastas, lambdaValue=30, w=0.05, **kw):
    records = []
    records.append("#	A	R	N	D	C	Q	E	G	H	I	L	K	M	F	P	S	T	W	Y	V")
    records.append(
        "Hydrophobicity	0.62	-2.53	-0.78	-0.9	0.29	-0.85	-0.74	0.48	-0.4	1.38	1.06	-1.5	0.64	1.19	0.12	-0.18	-0.05	0.81	0.26	1.08")
    records.append(
        "Hydrophilicity	-0.5	3	0.2	3	-1	0.2	3	0	-0.5	-1.8	-1.8	3	-1.3	-2.5	0	0.3	-0.4	-3.4	-2.3	-1.5")
    records.append(
        "SideChainMass	15	101	58	59	47	72	73	1	82	57	57	73	75	91	42	31	45	130	107	43")
    AA = ''.join(records[0].rstrip().split()[1:])
    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i
    AAProperty = []
    AAPropertyNames = []
    for i in range(1, len(records)):
        array = records[i].rstrip().split(
        ) if records[i].rstrip() != '' else None
        AAProperty.append([float(j) for j in array[1:]])
        AAPropertyNames.append(array[0])

    AAProperty1 = []
    for i in AAProperty:
        meanI = sum(i) / 20
        fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
        AAProperty1.append([(j - meanI) / fenmu for j in i])

    encodings = []
    header = []
    for aa in AA:
        header.append('Xc1.' + aa)
    for n in range(1, lambdaValue + 1):
        header.append('Xc2.lambda' + str(n))

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = []
        theta = []
        for n in range(1, lambdaValue + 1):
            theta.append(
                sum([Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1) for j in range(len(sequence) - n)]) / (
                    len(sequence) - n))
        myDict = {}
        for aa in AA:
            myDict[aa] = sequence.count(aa)
        code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
        code = code + [(w * j) / (1 + w * sum(theta)) for j in theta]
        encodings.append(code)
    return np.array(encodings, dtype=float), header


def APAAC(fastas, lambdaValue=30, w=0.05, **kw):
    records = []
    records.append("#	A	R	N	D	C	Q	E	G	H	I	L	K	M	F	P	S	T	W	Y	V")
    records.append(
        "Hydrophobicity	0.62	-2.53	-0.78	-0.9	0.29	-0.85	-0.74	0.48	-0.4	1.38	1.06	-1.5	0.64	1.19	0.12	-0.18	-0.05	0.81	0.26	1.08")
    records.append(
        "Hydrophilicity	-0.5	3	0.2	3	-1	0.2	3	0	-0.5	-1.8	-1.8	3	-1.3	-2.5	0	0.3	-0.4	-3.4	-2.3	-1.5")
    records.append(
        "SideChainMass	15	101	58	59	47	72	73	1	82	57	57	73	75	91	42	31	45	130	107	43")

    AA = ''.join(records[0].rstrip().split()[1:])
    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i
    AAProperty = []
    AAPropertyNames = []
    for i in range(1, len(records) - 1):
        array = records[i].rstrip().split(
        ) if records[i].rstrip() != '' else None
        AAProperty.append([float(j) for j in array[1:]])
        AAPropertyNames.append(array[0])

    AAProperty1 = []
    for i in AAProperty:
        meanI = sum(i) / 20
        fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
        AAProperty1.append([(j - meanI) / fenmu for j in i])

    encodings = []
    header = []
    for i in AA:
        header.append('Pc1.' + i)
    for j in range(1, lambdaValue + 1):
        for i in AAPropertyNames:
            header.append('Pc2.' + i + '.' + str(j))

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = []
        theta = []

        for n in range(1, lambdaValue + 1):
            for j in range(len(AAProperty1)):
                theta.append(sum([AAProperty1[j][AADict[sequence[k]]] * AAProperty1[j][AADict[sequence[k + n]]] for k in
                                  range(len(sequence) - n)]) / (len(sequence) - n))
        myDict = {}
        for aa in AA:
            myDict[aa] = sequence.count(aa)

        code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
        code = code + [w * value / (1 + w * sum(theta)) for value in theta]

        encodings.append(code)
    return np.array(encodings, dtype=float), header


def reducedACID(seq):
    def fcount(string, substr):
       count = 0
       pos = 0
       while(True):
           pos = string.find(substr, pos)
           if pos > -1:
               count = count + 1
               pos += 1
           else:
               break
       return count

    for count, fasta in enumerate(seq):
        sub = "akn"
        subsub = [it1+it2 for it1 in sub for it2 in sub]
        aalist = "ACDEFGHIKLMNPQRSTVWY"
        aasub = {}
        aasub["a"] = "DE"
        aasub["k"] = "KHR"
        aasub["n"] = "ACFGILMNPQSTVWY"

        seq1 = fasta[1]
        lenn = len(seq1)
        seq2 = seq1
        for key, value in aasub.items():
            for aa in value:
                seq2 = seq2.replace(aa, key)

        freq2 = {}
        for item in sub:
            freq2[item] = fcount(seq2, item)
        for item in subsub:
            freq2[item] = fcount(seq2, item)

        freq1 = {}
        for item in aalist:
            freq1[item] = fcount(seq1, item)

        feat = []
        for key, value in aasub.items():
            feat.append(freq2[key]/lenn)

        for item in aalist:
            for key, value in aasub.items():
                if item in value:
                    feat.append(freq1[item]/max(1, freq2[key]))
                    break

        for item in subsub:
            feat.append(freq2[item]/(freq2[item[0]]+1))

        feat = np.array(feat)
        feat = feat.reshape(1, len(feat))
        if count == 0:
            allfeat = feat
        else:
            allfeat = np.vstack((allfeat, feat))

    return allfeat


def reducedCHARGE(seq):
    def fcount(string, substr):
       count = 0
       pos = 0
       while(True):
           pos = string.find(substr, pos)
           if pos > -1:
               count = count + 1
               pos += 1
           else:
               break
       return count

    for count, fasta in enumerate(seq):
        sub = "qwe"
        subsub = [it1+it2 for it1 in sub for it2 in sub]
        aalist = "ACDEFGHIKLMNPQRSTVWY"
        aasub = {}
        aasub["q"] = "KR"
        aasub["w"] = "AVNCQGHILMFPSTWY"
        aasub["e"] = "DE"

        seq1 = fasta[1]
        lenn = len(seq1)
        seq2 = seq1
        for key, value in aasub.items():
            for aa in value:
                seq2 = seq2.replace(aa, key)

        freq2 = {}
        for item in sub:
            freq2[item] = fcount(seq2, item)
        for item in subsub:
            freq2[item] = fcount(seq2, item)

        freq1 = {}
        for item in aalist:
            freq1[item] = fcount(seq1, item)

        feat = []
        for key, value in aasub.items():
            feat.append(freq2[key]/lenn)

        for item in aalist:
            for key, value in aasub.items():
                if item in value:
                    feat.append(freq1[item]/max(1, freq2[key]))
                    break

        for item in subsub:
            feat.append(freq2[item]/(freq2[item[0]]+1))

        feat = np.array(feat)
        feat = feat.reshape(1, len(feat))
        if count == 0:
            allfeat = feat
        else:
            allfeat = np.vstack((allfeat, feat))

    return allfeat
