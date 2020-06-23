#!/usr/bin/env python
# _*_coding:utf-8_*_


import itertools
import pickle
import argparse
import os,sys,re
import numpy as np
from collections import Counter
from keras.models import load_model
import xgboost as xgb
import pandas as pd

def binary(sequences):
    AA = 'ACGU'
    binary_feature = []
    for seq in sequences:
        binary = []
        for aa in seq:
            for aa1 in AA:
                tag = 1 if aa == aa1 else 0
                binary.append(tag)
        binary_feature.append(binary)
    return binary_feature

def CKSNAP(sequences):
    K=2
    cksnap_feature = []
    AA = 'ACGU'
    AApairs = []
    for aa1 in AA:
        for aa2 in AA:
            AApairs.append(aa1 + aa2)
    for seq in sequences:
        cksnap = []
        l = len(seq)
        for k in range(0, K + 1):
            record_dict = {}
            for i in AApairs:
                record_dict[i] = 0

            sum = 0
            for index1 in range(l):
                index2 = index1 + k + 1
                if index1 < l and index2 < l:
                    record_dict[seq[index1] + seq[index2]] = record_dict[seq[index1] + seq[index2]] + 1
                    sum = sum + 1

            for pair in AApairs:
                cksnap.append(record_dict[pair] / sum)
        cksnap_feature.append(cksnap)
    return cksnap_feature


def NCP(sequences):
    chemical_property = {
        'A': [1, 1, 1],
        'C': [0, 1, 0],
        'G': [1, 0, 0],
        'U': [0, 0, 1], }
    ncp_feature = []
    for seq in sequences:
        ncp = []
        for aaindex, aa in enumerate(seq):
            ncp = ncp + chemical_property.get(aa, [0, 0, 0])
        ncp_feature.append(ncp)
    return ncp_feature


def ND(sequences):
    nd_feature = []
    for seq in sequences:
        nd = []
        for aaindex, aa in enumerate(seq):
            nd.append(seq[0: aaindex + 1].count(seq[aaindex]) / (aaindex + 1))
        nd_feature.append(nd)
    return nd_feature



def ENAC(sequences):
    AA = 'ACGU'
    enac_feature = []
    window = 2
    for seq in sequences:
        l = len(seq)
        enac= []
        for i in range(0, l):
            if i < l and i + window <= l:
                count = Counter(seq[i:i + window])
                for key in count:
                    count[key] = count[key] / len(seq[i:i + window])
                for aa in AA:
                    enac.append(count[aa])
        enac_feature.append(enac)
    return enac_feature






myDiIndex = {
    'AA': 0, 'AC': 1, 'AG': 2, 'AU': 3,
    'CA': 4, 'CC': 5, 'CG': 6, 'CU': 7,
    'GA': 8, 'GC': 9, 'GG': 10, 'GU': 11,
    'UA': 12, 'UC': 13, 'UG': 14, 'UU': 15
}



baseSymbol = 'ACGU'


def get_kmer_frequency(sequence, kmer):
    frequency = {}
    for pep in [''.join(i) for i in list(itertools.product(baseSymbol, repeat=kmer))]:
        #itertools.product(A,B),返回A，B中元素的笛卡尔积的元祖，product(A,repeat=4)的含义与product(A,A,A,A)的含义相同。
        frequency[pep] = 0
    for i in range(len(sequence) - kmer + 1):
        frequency[sequence[i: i + kmer]] = frequency[sequence[i: i + kmer]] + 1
    for key in frequency:
        frequency[key] = frequency[key] / (len(sequence) - kmer + 1)
    return frequency #返回的是一个字典‘AAAAA’：0.34




def correlationFunction_type2(pepA, pepB, myIndex, myPropertyName, myPropertyValue):
    CC = 0
    for p in myPropertyName:
        CC = CC + float(myPropertyValue[p][myIndex[pepA]]) * float(myPropertyValue[p][myIndex[pepB]])
    return CC




def get_theta_array_type2(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence,i):
    fixkmer = i
    thetaArray = []
    for tmpLamada in range(lamadaValue):
        for p in myPropertyName:
            theta = 0
            for i in range(len(sequence) - tmpLamada - fixkmer):
                theta = theta + correlationFunction_type2(sequence[i:i + fixkmer],
                                                          sequence[i + tmpLamada + 1: i + tmpLamada + 1 + fixkmer],
                                                          myIndex,
                                                          [p], myPropertyValue)
            thetaArray.append(theta / (len(sequence) - tmpLamada - fixkmer))
    return thetaArray




def SCPseDNC(sequences):
    property_name = ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']
    dataFile = 'dirnaPhyche.data'
    with open(dataFile,'rb') as f:
        property_value = pickle.load(f)
    lamada =  20#20
    weight =  0.9#0.9
    kmer_index = myDiIndex
    SCPseDNC_feature = []
    for i in sequences:
        code = []
        dipeptideFrequency = get_kmer_frequency(i, 2)
        thetaArray = get_theta_array_type2(kmer_index, property_name, property_value, lamada, i,2)
        for pep in sorted(kmer_index.keys()):
            code.append(dipeptideFrequency[pep] / (1 + weight * sum(thetaArray)))
        for k in range(17, 16 + lamada * len(property_name) + 1):
            code.append((weight * thetaArray[k - 17]) / (1 + weight * sum(thetaArray)))
        SCPseDNC_feature.append(code)
    return SCPseDNC_feature











def read_fasta(inputfile):
    if os.path.exists(inputfile) == False:
        print('Error: file " %s " does not exist.' % inputfile)
        sys.exit(1)
    with open(inputfile) as f:
        record = f.readlines()
    if re.search('>',record[0]) == None:
        print('Error: the input file " %s " must be fasta format!' % inputfile)
        sys.exit(1)

    data = {}
    for line in record:
        if line.startswith('>'):
            name = line.replace('>','').split('\n')[0]
            data[name] = ''
        else:
            data[name] += line.replace('\n','')
    return data




def extract_features(data):
    sequences = data
    basic1 = np.array(binary(sequences))
    basic2 = np.array(CKSNAP(sequences))
    basic3 = np.array(ENAC(sequences))
    basic4 = np.array(NCP(sequences))
    basic5 = np.array(ND(sequences))
    basic6 = np.array(SCPseDNC(sequences))

    feature_vector = np.concatenate((basic1,basic2,basic3,basic4,basic5,basic6), axis=1)
    return feature_vector





def predict_m7G(data,outputfile):
    bs = xgb.Booster({'nthread': 4})  # init model
    bs.load_model('xgb.model')

    vector = extract_features(data.values())
    df = pd.DataFrame(vector)
    feature_names = []
    for i in range(1, len(df.columns)):
        feature_names.append(df.columns[i])
    ppp = df[feature_names]
    ttt = xgb.DMatrix(ppp)

    predictions = bs.predict(ttt)
    probability = ['%.5f' % float(i) for i in predictions]
    name = list(data.keys())
    seq = list(data.values())
    with open(outputfile,'w') as f:
        for i in range(len(data)):
            if float(probability[i]) > 0.5:
                f.write(probability[i]+'*' + '\t')
                f.write(name[i] + '\t')
                f.write(seq[i] + '\n')
            else:
                f.write(probability[i] + '\t')
                f.write(name[i] + '\t')
                f.write(seq[i] + '\n')
    return None



import datetime
import time

def self_train(data):
    full_name = data.keys()
    Y = []
    for i in full_name:
        ii = i.split(' ')[-1]
        Y.append(ii)
    Y = np.array(Y)
    X = extract_features(data.values())
    model = xgb.XGBClassifier(n_estimators=1000, max_depth=3, learning_rate=0.2, gamma=0.001)
    model.fit(X, Y)

    t = time.time()
    D =  datetime.date.today()
    file_name = str(D) + '+' +str(round(t))
    model.save_model(file_name + '.model')
    return None



def main():
    parser = argparse.ArgumentParser(description='XG-m7G: a XGBoost-based approach to identify N7-methylguanosine sites')
    parser.add_argument('--input',dest='inputfile',type=str,required=True,help='RNA sequences to be predicted or self-training in fasta format.'
                                                                               'If you want to self train model, please put the label after the fasta name.'
                                                                               'For example:'
                                                                               '>example1 1'
                                                                               'AAGAACAGGAGCGAGAGAAGGAGAGGGAAAAAGACAGAGAG'
                                                                               '>example2 0'
                                                                               'CAGCGAGUUCGGUUGCGCGUGACGCACCGGGUGGGAGCGGA'
                        )
    parser.add_argument('--purpose', dest='purpose', type=str, required=True,
                        help='you can choose "predict" or "self-training" to indicate your purpose')
    parser.add_argument('--output',dest='outputfile',type=str,required=False,help='Save the prediction  results in csv format.')
    args = parser.parse_args()

    inputfile = args.inputfile
    aim = args.purpose
    outputfile = args.outputfile



    if aim == 'predict':
        data = read_fasta(inputfile)
        if outputfile != None:
            predict_m7G(data, outputfile)
            print(
                'output are saved in ' + outputfile + ', and those with probability greater than 0.5 are marked with *')
        else:
            default_output = 'output'
            predict_m7G(data, default_output)
            print(
                'output are saved in the current directory' + ', and those with probability greater than 0.5 are marked with *')

    elif aim == 'self-training':
        data = read_fasta(inputfile)
        self_train(data)
        print('self-training model is saved in the directory')


if __name__ == "__main__":
    main()


