import numpy as np
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
import math
import operator
import re
import os
from random import random
import shutil
import Cluster.clustering
from Cluster.clustering import *

datasets = ["ItalyPowerDemand"]

centroids_L = []
results_L = []
SSR_k = []
SSR_info = []
SSR_k_initial = 1

writeRes = False
isSaveCentroids = False
getTime = False
train_data = None

change_label = True

MAX_K = 5

def predictLsAcc(predict_dict, data_size, true_label):
    predict_ls = []
    for i in range(data_size):
        dict1 = predict_dict[i]

        label_max_cnt = -1000000
        label_max = 100000
        for label in dataLabel:
            if dict1[label] > label_max_cnt:
                label_max_cnt = dict1[label]
                label_max = label
        predict_ls.append(label_max)

    trueCnt = 0
    for i in range(0, len(predict_ls)):
        if true_label[i] == predict_ls[i]:
            trueCnt += 1
    acc = trueCnt / len(true_label)

    return acc

def getKpredictLsAcc(results_l, data_size, data_label):
    Allfeature_predict_dict = []
    for i in range(data_size):
        tmp_dict = {}
        for datalabel in dataLabel:
            tmp_dict[datalabel] = 0
        Allfeature_predict_dict.append(tmp_dict)

    SSR_len = len(SSR_k)
    # print(SSR_k)
    for i in range(SSR_len):
        if len(results_l[i]) ==0:
            continue
        f_res_dict = results_l[i][SSR_k[i]]
        for j in range(data_size):
            tmp_dict = f_res_dict[j]
            for datalabel in dataLabel:
                Allfeature_predict_dict[j][datalabel] += tmp_dict[datalabel]

    acc = predictLsAcc(Allfeature_predict_dict, data_size, data_label)
    return acc

RISE_transform_type = ["PS","ACF"]
data_root = "1. Feature candidates/FCCA/data/"

if __name__ == "__main__":

    root = "1. Feature candidates/FCCA/result/"
    res_file = root + "accuracy.csv"

    All_acc = []
    ALL_featureNum = []

    for datasetId, dataset in enumerate(datasets):

        print("=========== dataset: " + dataset + "==============")
        data_root_train = data_root + dataset + "/" + dataset + "_TRAIN.tsv"
        train_data = pd.read_csv(data_root_train, delimiter='\t', header=None)

        data_root_validate = data_root + dataset + "/" + dataset + "_VALIDATE.tsv"
        validate_data = pd.read_csv(data_root_validate, delimiter='\t', header=None)
        validate_label = list(validate_data[0])
        validatedata_size = len(validate_label)

        initial(train_data, change_label, validatedata_size, validate_label)

        data_root_test = data_root + dataset + "/" + dataset + "_TEST.tsv"
        test_data = pd.read_csv(data_root_test, delimiter='\t', header=None)
        test_label = list(test_data[0])
        testdata_size = len(test_label)

        dataLabel = list(np.unique(test_label))
        labelTypeCnt = len(dataLabel)

        topN_path = root + dataset + "/sfa_cluster"

        SAX_cluster_num = [i for i in range(1, MAX_K + 1)]
        SFA_cluster_num = [i for i in range(1, MAX_K + 1)]
        RISE_cluster_num = [i for i in range(1, MAX_K + 1)]

        SSR_k = []
        SSR_info = []
        centroids_L = []
        results_L = []

        SSR_data_num = []

        SAXfeature_df = None
        SAXtrain_df = None
        SAXtest_df = None
        SAX_featurelist = []
        sax_result_file = topN_path + "/SAXfeature_topN.csv"
        if os.path.exists(sax_result_file) and os.path.getsize(sax_result_file) > 0:
            SAXfeature_df = pd.read_csv(sax_result_file, header=None)

            sax_validate_file = topN_path + "/SAX_validate_topN.csv"
            SAXvalidate_df = pd.read_csv(sax_validate_file, header=None)
            initial_SAXdf(SAXvalidate_df)

            sax_test_file = topN_path + "/SAX_test_topN.csv"
            SAXtest_df = pd.read_csv(sax_test_file, header=None)

            for label in dataLabel:
                sax_featureLs_label = []
                sax_featureDf_tmp = SAXfeature_df[SAXfeature_df[2] == label]
                sax_featureDf_tmp_sort = sax_featureDf_tmp.sort_values(by=[3, 6], ascending=[False, True])
                for idx, row in sax_featureDf_tmp_sort.iterrows():
                    sax_featureLs_label.append(list(row))

                if len(sax_featureLs_label) == 0:
                    centroids_L.append({})
                    results_L.append({})
                    SSR_k.append(0)
                    SSR_info.append("SAX_" + str(label))
                    SSR_data_num.append(0)
                    continue
                SSR_data_num.append(len(sax_featureLs_label))

                sax_centroids_info = run_CURE(sax_featureLs_label, SAX_cluster_num, calDsitance_SAX,
                                              cal_dis_between_clusters)
                centroids_L.append(sax_centroids_info)

                sax_centroids_result = {}
                for sax_k in sax_centroids_info:
                    sax_k_info = sax_centroids_info[sax_k]
                    sax_k_result = predictByCentroids(sax_k_info, SAXvalidate_df, validatedata_size, dataLabel)
                    sax_centroids_result[sax_k] = sax_k_result
                results_L.append(sax_centroids_result)

                SSR_k.append(SSR_k_initial)
                SSR_info.append("SAX_" + str(label))
        else:
            for label in dataLabel:
                SSR_k.append(0)
                SSR_info.append("SAX_" + str(label))
                SSR_data_num.append(0)
                centroids_L.append({})
                results_L.append({})

        SFAfeature_df = None
        SFAtrain_df = None
        SFAtest_df = None
        SFA_featurelist = []
        sfa_result_file = topN_path + "/SFAfeature_topN.csv"
        if os.path.exists(sfa_result_file) and os.path.getsize(sfa_result_file) > 0:
            SFAfeature_df = pd.read_csv(sfa_result_file, header=None)

            sfa_validate_file = topN_path + "/SFA_validate_topN.csv"
            SFAvalidate_df = pd.read_csv(sfa_validate_file, header=None)
            initial_SFAdf(SFAvalidate_df)

            sfa_test_file = topN_path + "/SFA_test_topN.csv"
            SFAtest_df = pd.read_csv(sfa_test_file, header=None)

            for label in dataLabel:
                sfa_featureLs_label = []
                sfa_featureDF_tmp = SFAfeature_df[SFAfeature_df[2] == label]
                sfa_featureDF_tmp_sort = sfa_featureDF_tmp.sort_values(by=[3, 7], ascending=[False, True])
                for idx, row in sfa_featureDF_tmp_sort.iterrows():
                    sfa_featureLs_label.append(list(row))

                if len(sfa_featureLs_label) == 0:
                    centroids_L.append({})
                    results_L.append({})
                    SSR_k.append(0)
                    SSR_info.append("SFA_" + str(label))
                    SSR_data_num.append(0)
                    continue
                SSR_data_num.append(len(sfa_featureLs_label))

                sfa_centroids_info = run_CURE(sfa_featureLs_label, SFA_cluster_num, calDsitance_SFA,
                                              cal_dis_between_clusters)
                centroids_L.append(sfa_centroids_info)

                sfa_centroids_result = {}
                for sfa_k in sfa_centroids_info:
                    sfa_k_info = sfa_centroids_info[sfa_k]
                    sfa_k_result = predictByCentroids(sfa_k_info, SFAvalidate_df, validatedata_size, dataLabel)
                    sfa_centroids_result[sfa_k] = sfa_k_result
                results_L.append(sfa_centroids_result)

                SSR_k.append(SSR_k_initial)
                SSR_info.append("SFA_" + str(label))
        else:
            for label in dataLabel:
                SSR_k.append(0)
                SSR_info.append("SFA_" + str(label))
                SSR_data_num.append(0)
                centroids_L.append({})
                results_L.append({})

        RISEfeature_df = None
        RISEtrain_df = None
        RISEtest_df = None
        RISE_featurelist = []
        rise_result_file = topN_path + "/RISEfeature_topN.csv"
        if os.path.exists(rise_result_file) and os.path.getsize(rise_result_file) > 0:
            RISEfeature_df = pd.read_csv(rise_result_file, header=None)

            rise_validate_file = topN_path + "/RISE_validate_topN.csv"
            RISEvalidate_df = pd.read_csv(rise_validate_file, header=None)
            initial_RISEdf(RISEvalidate_df)

            rise_test_file = topN_path + "/RISE_test_topN.csv"
            RISEtest_df = pd.read_csv(rise_test_file, header=None)

            for label in dataLabel:
                rise_featureDF_tmp = RISEfeature_df[RISEfeature_df[2] == label]
                rise_featureDF_label = rise_featureDF_tmp.sort_values(by=3, ascending=False)

                for transform_type in RISE_transform_type:
                    featureDF_riseType = rise_featureDF_label[rise_featureDF_label[9] == transform_type]
                    riseType_num = featureDF_riseType.shape[0]
                    if riseType_num == 0:
                        continue

                    featureLs_riseType = []
                    for idx, rows in featureDF_riseType.iterrows():
                        featureLs_riseType.append(list(rows))
                    SSR_data_num.append(len(featureLs_riseType))

                    rise_centroids_info = run_CURE(featureLs_riseType, RISE_cluster_num, calDsitance_RISE,
                                                   cal_dis_between_clusters)
                    centroids_L.append(rise_centroids_info)

                    rise_centroids_result = {}
                    for rise_k in rise_centroids_info:
                        rise_k_info = rise_centroids_info[rise_k]
                        rise_k_result = predictByCentroids(rise_k_info, RISEvalidate_df, validatedata_size,
                                                           dataLabel)
                        rise_centroids_result[rise_k] = rise_k_result
                    results_L.append(rise_centroids_result)

                    SSR_k.append(SSR_k_initial)
                    SSR_info.append(transform_type + "_" + str(label))

        acc = getKpredictLsAcc(results_L, validatedata_size, validate_label)

        SSR_len = len(SSR_k)
        pre_acc = -10
        pre_SSR_k = None

        while acc >= pre_acc:
            pre_acc = acc
            pre_SSR_k = SSR_k.copy()
            iter_acc = []

            flag = False
            for i in range(SSR_len):
                if SSR_k[i] > 0 and SSR_k[i] < MAX_K and SSR_k[i] < SSR_data_num[i]:
                    flag = True
                    break
            if flag is False:
                break

            for i in range(SSR_len):
                if SSR_k[i] == 0 or SSR_k[i] == MAX_K or SSR_k[i] == SSR_data_num[i]:
                    iter_acc.append(-1)
                    continue
                SSR_k[i] = SSR_k[i] + 1

                results_tmp_L = []
                for j in range(SSR_len):
                    if SSR_k[j] == 0:
                        results_tmp_L.append({})
                        continue
                    results_tmp = results_L[j][SSR_k[j]]
                    results_tmp_L.append(results_tmp)

                predict_dict_ls_tmp = []
                for k in range(validatedata_size):
                    tmp_dict = {}
                    for datalabel in dataLabel:
                        tmp_dict[datalabel] = 0
                    predict_dict_ls_tmp.append(tmp_dict)

                for k in range(SSR_len):
                    if SSR_k[k] == 0:
                        continue
                    f_res_dict = results_tmp_L[k]
                    for j in range(validatedata_size):
                        tmp_dict = f_res_dict[j]
                        for datalabel in dataLabel:
                            predict_dict_ls_tmp[j][datalabel] += tmp_dict[datalabel]

                validate_acc = predictLsAcc(predict_dict_ls_tmp, validatedata_size, validate_label)
                iter_acc.append(validate_acc)

                SSR_k[i] = SSR_k[i] - 1

            best_pos = np.argmax(iter_acc)
            SSR_k[best_pos] = SSR_k[best_pos] + 1
            acc = np.max(iter_acc)

        SSR_k = pre_SSR_k.copy()

        results_test_L = []
        for i in range(labelTypeCnt):
            if SSR_k[i] == 0:
                results_test_L.append([])
                continue
            SAX_centroids_info = centroids_L[i][SSR_k[i]]
            SAX_result = predictByCentroids(SAX_centroids_info, SAXtest_df, testdata_size, dataLabel)
            results_test_L.append(SAX_result)

        for i in range(labelTypeCnt):
            if SSR_k[i + labelTypeCnt] == 0:
                results_test_L.append([])
                continue
            SFA_centroids_info = centroids_L[i + labelTypeCnt][SSR_k[i + labelTypeCnt]]
            SFA_result = predictByCentroids(SFA_centroids_info, SFAtest_df, testdata_size, dataLabel)
            results_test_L.append(SFA_result)

        for i in range(labelTypeCnt * 2, SSR_len):
            if SSR_k[i] == 0:
                results_test_L.append([])
                continue
            RISE_centroids_info = centroids_L[i][SSR_k[i]]
            RISE_result = predictByCentroids(RISE_centroids_info, RISEtest_df, testdata_size, dataLabel)
            results_test_L.append(RISE_result)

        predict_dict_ls = []
        for i in range(testdata_size):
            tmp_dict = {}
            for datalabel in dataLabel:
                tmp_dict[datalabel] = 0
            predict_dict_ls.append(tmp_dict)

        SSR_len = len(SSR_k)
        for i in range(SSR_len):
            if SSR_k[i] == 0:
                continue
            f_res_dict = results_test_L[i]
            for j in range(testdata_size):
                tmp_dict = f_res_dict[j]
                for datalabel in dataLabel:
                    predict_dict_ls[j][datalabel] += tmp_dict[datalabel]

        test_acc = predictLsAcc(predict_dict_ls, testdata_size, test_label)

        All_acc.append(test_acc)
        ALL_featureNum.append(np.sum(SSR_k))

        print("test acc:", test_acc)
        print("featureNum: ", np.sum(SSR_k))
        print("best_cluster_K: ", MAX_K)

        if isSaveCentroids:
            centroids_path = topN_path + "/centroids"
            if os.path.exists(centroids_path):
                shutil.rmtree(centroids_path)
                os.mkdir(centroids_path)
            elif not os.path.exists(centroids_path):
                os.mkdir(centroids_path)

            best_SSR_len = len(best_SSR)
            best_centroids_l = []
            for i in range(best_SSR_len):
                if best_SSR[i] == 0:
                    continue
                centroids_info = centroids_L[i][SSR_k[i]]
                for cc in centroids_info.centroids:
                    best_centroids_l.append(cc)
            best_centroids_df = pd.DataFrame(best_centroids_l)

            centroids_file = centroids_path + "/centroids.csv"
            best_centroids_df.to_csv(centroids_file, header=None, index=None)

    res = []
    res.append(datasets)
    res.append(All_acc)
    res.append(ALL_featureNum)

    res_df = pd.DataFrame(res).T

    if writeRes:
        if os.path.exists(res_file):
            os.remove(res_file)
        res_df.to_csv(res_file, header=None, index=None)