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

train_data = None
change_label = True
validate_label = None
validatedata_size = 0

SAXvalidate_df = pd.DataFrame()
SFAvalidate_df = pd.DataFrame()
RISEvalidate_df = pd.DataFrame()

def initial(data, change_label1, valDataSize, valLabel):
    global train_data
    train_data = data
    global change_label
    change_label = change_label1
    global validate_label
    validate_label = valLabel
    global validatedata_size
    validatedata_size = valDataSize

def initial_SAXdf(SAX_df):
    global SAXvalidate_df
    SAXvalidate_df = SAX_df.copy()

def initial_SFAdf(SFA_df):
    global SFAvalidate_df
    SFAvalidate_df = SFA_df.copy()

def initial_RISEdf(RISE_df):
    global RISEvalidate_df
    RISEvalidate_df = RISE_df.copy()

def cal_dis_between_clusters(dist_cluster_origin, cluster1, cluster2):
    cluster1_idx = cluster1.index
    cluster2_idx = cluster2.index
    dis_max = 0
    for idx1 in cluster1_idx:
        for idx2 in cluster2_idx:
            if dist_cluster_origin[idx1][idx2] > dis_max:
                dis_max = dist_cluster_origin[idx1][idx2]

    return dis_max

def calDsitance_SFA(sfa_feature1, sfa_feature2):
    dis = 0

    # 1. 计算SFA word之间生成的距离
    sfa_word1 = sfa_feature1[7]
    sfa_word2 = sfa_feature2[7]

    sfa_word_dis = 0
    sfa_len = min(len(sfa_word1), len(sfa_word2))
    for i in range(sfa_len):
        # sfa_word_dis += sfa_weight[i] * (abs(ord(sfa_word1[i])-ord(sfa_word2[i])))
        sfa_word_dis += (abs(ord(sfa_word1[i]) - ord(sfa_word2[i])))

    dis += sfa_word_dis

    # 3. 来源时间序列的长度
    len1 = int(sfa_feature1[8])
    len2 = int(sfa_feature2[8])
    len_max = max(len1, len2)
    len_min = min(len1, len2)
    len_factor = pow(2, log(len_min/len_max))

    return dis*len_factor

def calDsitance_SAX(featureX, featureY):

    featureX_tsId = int(featureX[7])
    featureY_tsId = int(featureY[7])

    featureX_ts_st = int(featureX[11])
    featureY_ts_st = int(featureY[11])

    featureX_len = int(featureX[8])
    featureY_len = int(featureY[8])

    featureX_ts = list(train_data.iloc[featureX_tsId, 1:])
    featureY_ts = list(train_data.iloc[featureY_tsId, 1:])

    featureX_origin = featureX_ts[featureX_ts_st: featureX_ts_st+featureX_len]
    featureY_origin = featureY_ts[featureY_ts_st: featureY_ts_st+featureY_len]

    if len(featureX_origin)>len(featureY_origin):
        tmp = featureY_origin
        featureY_origin = featureX_origin
        featureX_origin = tmp

    # featureX_origin一定小于featureY_origin
    offset = len(featureY_origin)-len(featureX_origin) + 1
    dis_min = float('inf')
    for i in range(offset):
        dis = 0
        # featureY_origin = min_max_scaler.fit_transform(np.array(featureY_tmp[i:i+len(featureX_origin)]).reshape(-1,1)).reshape(1,-1)[0].tolist()
        for j in range(len(featureX_origin)):
            dis += (featureY_origin[i+j]-featureX_origin[j])*(featureY_origin[i+j]-featureX_origin[j])
        dis = np.sqrt(dis)
        dis = dis/np.sqrt(len(featureX_origin))
        # dis = dis / len(featureX_origin)
        if dis < dis_min:
            dis_min = dis

    return dis_min

def calDsitance_RISE(rise_feature1, rise_feature2):
    dis = 0
    dis += abs(rise_feature1[7]-rise_feature2[7]) + abs(rise_feature1[8]-rise_feature2[8]) + abs(rise_feature1[6]-rise_feature2[6])
    return dis

def getFeatureValacc(centroid):
    feature_label = int(centroid[2])
    feature_Id = int(centroid[0])
    feature_threshold = int(float(centroid[4]))
    feature_Sign = int(centroid[5])
    feature_type = centroid[1]

    test_tmp = None
    if feature_type == 'SAX':
        test_tmp = list(SAXvalidate_df[SAXvalidate_df[0] == feature_Id].values[0])[1:]
    elif feature_type == 'SFA':
        test_tmp = list(SFAvalidate_df[SFAvalidate_df[0] == feature_Id].values[0])[1:]
    else:
        test_tmp = list(RISEvalidate_df[RISEvalidate_df[0] == feature_Id].values[0])[1:]

    feature_predict_ls = []
    for tmp in test_tmp:
        if feature_Sign == 1:
            if tmp >= feature_threshold:
                feature_predict_ls.append(feature_label)
            else:
                feature_predict_ls.append(feature_label + 100000)
        else:
            if tmp < feature_threshold:
                feature_predict_ls.append(feature_label)
            else:
                feature_predict_ls.append(feature_label + 100000)

    acc_cnt = 0
    for i in range(validatedata_size):
        if feature_predict_ls[i] == feature_label and validate_label[i] == feature_label:
            acc_cnt += 1
        elif feature_predict_ls[i] != feature_label and validate_label[i] != feature_label:
            acc_cnt += 1
    acc = acc_cnt/validatedata_size

    return acc

class CureCluster:
    def __init__(self, id__, center__):
        self.points = center__
        self.center = center__
        self.centerId = id__
        self.index = [id__]

    def __repr__(self):
        return self.index

    # Computes and stores the centroid of this cluster, based on its points
    def compute_centroid_by_dissum(self, dist_cluster_origin):
        p2p_sumdis = []
        pts_num = len(self.index)
        # print(self.points)
        for i in range(pts_num):
            dis_tmp = 0
            x_id = self.index[i]
            for j in range(pts_num):
                if i == j:
                    continue
                y_id = self.index[j]
                dis_tmp+=dist_cluster_origin[x_id][y_id]
            p2p_sumdis.append(dis_tmp)
        new_center_id = p2p_sumdis.index(min(p2p_sumdis))
        self.center = self.points[new_center_id]
        self.centerId = self.index[new_center_id]

    def compute_centroid_by_infogain(self):
        pts_infogain = []
        pts_num = len(self.index)
        # print(self.points)
        for i in range(pts_num):
            pts_infogain.append(float(self.points[i][3]))
        new_center_id = pts_infogain.index(max(pts_infogain))
        self.center = self.points[new_center_id]
        self.centerId = self.index[new_center_id]

    def compute_centroid_by_infogainAndPreacc(self):
        pts_w = []
        pts_num = len(self.index)
        # print(self.points)
        for i in range(pts_num):
            g_tmp = float(self.points[i][3])
            acc_tmp = getFeatureValacc(self.points[i])
            pts_w.append(g_tmp * acc_tmp)
        new_center_id = pts_w.index(max(pts_w))
        self.center = self.points[new_center_id]
        self.centerId = self.index[new_center_id]

    def compute_centroid_by_infogainThenPreacc(self):
        pts_infogain = []
        pts_preacc = []
        pts_num = len(self.index)
        # print(self.points)
        for i in range(pts_num):
            g_tmp = float(self.points[i][3])
            acc_tmp = getFeatureValacc(self.points[i])
            pts_infogain.append(g_tmp)
            pts_preacc.append(acc_tmp)

        max_infogain = -1
        max_id = 0
        max_preacc = -1
        for ki, kk in enumerate(pts_infogain):
            if kk > max_infogain:
                max_id = ki
                max_infogain = kk
                max_preacc = pts_preacc[ki]
            elif kk == max_infogain:
                if max_preacc < pts_preacc[ki]:
                    max_id = ki
                    max_preacc = pts_preacc[ki]

        new_center_id = max_id
        self.center = self.points[new_center_id]
        self.centerId = self.index[new_center_id]

    def dist_center(self, clust, dist):
        return dist(self.center, clust.center)

    # Merges this cluster with the given cluster, recomputing the centroid and the representative points.
    def merge_with_cluster(self, clust, dist_cluster_origin):

        self.points = np.vstack((self.points, clust.points)).tolist()
        self.index.extend(clust.index)
        # self.compute_centroid_by_infogain()
        self.compute_centroid_by_infogainAndPreacc()
        # self.compute_centroid_by_infogainThenPreacc()

class clusterInfo:
    # 聚类中心，每个聚类中的特征个数，聚类中心信息增益，聚类中心权重
    def __init__(self, centroids, clu_num, centroids_infogain):
        self.centroids = centroids
        self.clu_num = clu_num
        self.centroids_infogain = centroids_infogain
        self.centroids_w = [b*log(a+1) for a,b in zip(clu_num,centroids_infogain)]

def run_CURE(data, cluster_k, dist, cal_dis_between_clusters):
    k_centroids_dict = {}

    # Initialization
    clusters = []
    num_cluster = len(data)
    num_Pts = len(data)
    dist_cluster = np.ones([len(data), len(data)])* np.inf

    for id_point in range(len(data)):
        new_clust = CureCluster(id_point, data[id_point])
        clusters.append(new_clust)

    for row in range(0, num_Pts):
        for col in range(0, row):
            dist_cluster[row][col] = dist_cluster[col][row] = dist(clusters[row].center, clusters[col].center)

    dist_cluster_origin = dist_cluster.copy()

    if num_cluster == 1:
        k_centroids_dict[num_cluster] = clusterInfo([data[0]], [1], [data[0][3]])
        return k_centroids_dict

    if num_cluster in cluster_k:
        infogain_tmp = []
        for d in data:
            infogain_tmp.append(d[3])
        k_centroids_dict[num_cluster] = clusterInfo(data, [1]*len(data), infogain_tmp)

    while num_cluster > 1:

        # Find a pair of closet clusters
        min_index = np.where(dist_cluster == np.min(dist_cluster))
        min_index1 = min_index[0][0]
        min_index2 = min_index[1][0]

        # Merge
        clusters[min_index1].merge_with_cluster(clusters[min_index2], dist_cluster_origin)

        # Update the distCluster matrix
        for i in range(0, min_index1):
            dist_cluster[min_index1, i] = cal_dis_between_clusters(dist_cluster_origin, clusters[i],
                                                                   clusters[min_index1])
        for i in range(min_index1 + 1, num_cluster):
            dist_cluster[i, min_index1] = cal_dis_between_clusters(dist_cluster_origin, clusters[i],
                                                                   clusters[min_index1])
        dist_cluster[min_index1, min_index1] = inf

        # print("删除聚类：", clusters[min_index2].index)
        # Delete the merged cluster and its disCluster vector.
        dist_cluster = np.delete(dist_cluster, min_index2, axis=0)
        dist_cluster = np.delete(dist_cluster, min_index2, axis=1)
        del clusters[min_index2]
        num_cluster = num_cluster - 1

        if num_cluster in cluster_k:
            centroids = []
            centroids_infogain = []
            clu_num = []
            for i in range(0, len(clusters)):
                centroids.append(clusters[i].center)
                clu_num.append(len(clusters[i].index))
                centroids_infogain.append(float(clusters[i].center[3]))
            k_centroids_dict[num_cluster] = clusterInfo(centroids, clu_num, centroids_infogain)

    return k_centroids_dict

def predictByCentroids(centroidsInfo, test_df, testdata_size, dataLabel):
    featureType_dic = []  # 该类特征对每个数据的预测
    for i in range(testdata_size):
        tmp_dict = {}
        for datalabel in dataLabel:
            tmp_dict[datalabel] = 0
        featureType_dic.append(tmp_dict)

    centroids_ls = centroidsInfo.centroids
    centroids_w = centroidsInfo.centroids_w
    centroids_predict_ls = []
    for idx, centroid in enumerate(centroids_ls):
        feature_label = int(centroid[2])
        feature_Id = int(centroid[0])
        feature_threshold = int(float(centroid[4]))
        feature_Sign = int(centroid[5])

        test_tmp = list(test_df[test_df[0] == feature_Id].values[0])[1:]
        feature_predict_ls = []
        for tmp in test_tmp:
            if feature_Sign == 1:
                if tmp >= feature_threshold:
                    feature_predict_ls.append(feature_label)
                else:
                    feature_predict_ls.append(feature_label + 100000)
            else:
                if tmp < feature_threshold:
                    feature_predict_ls.append(feature_label)
                else:
                    feature_predict_ls.append(feature_label + 100000)
        centroids_predict_ls.append(feature_predict_ls)

    centroids_predict_df = pd.DataFrame(centroids_predict_ls)
    for tid in centroids_predict_df.columns:
        predict_ls = list(centroids_predict_df[tid])
        predict_dict = featureType_dic[tid]

        for fid, pval in enumerate(predict_ls):
            if change_label:
                if pval > 50000:
                    predict_dict[pval - 100000] = predict_dict[pval - 100000] - centroids_w[fid]
                else:
                    predict_dict[pval] = predict_dict[pval] + centroids_w[fid]
            else:
                if pval > 50000:
                    continue
                else:
                    predict_dict[pval] = predict_dict[pval] + centroids_w[fid]
        featureType_dic[tid] = predict_dict

    return featureType_dic