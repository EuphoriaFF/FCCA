import numpy as np
import pandas as pd
import math
import os
import shutil

datasets = ["ItalyPowerDemand"]

choose_dataset = 0
choose_compare = 3

if choose_compare == 3:
    pp = 0.2

selectPerNum = 5
update_topN = True
data_types = ["fuzzy"]

change_label = True

if __name__ == "__main__":

    root = "1. Feature candidates/FCCA/result/"
    best_topN_ls = []

    for dataset in datasets:
        print("=========== dataset: " + dataset + "==============")
        data_root = "1. Feature candidates/FCCA/data/" + dataset + "/" + dataset + "_VALIDATE.tsv"
        test_data = pd.read_csv(data_root, delimiter='\t', header=None)
        test_label = test_data[0]
        dataLabel = list(np.unique(test_label))
        labelTypeCnt = len(dataLabel)
        testdata_size = len(test_label)

        selectPerNum_label = selectPerNum

        for data_type in data_types:
            print("=========== data_type: " + data_type + "==============")

            topN_acc = []

            feature_filename = root + dataset + "/ALLfeature_" + data_type + ".csv"
            feature_df = pd.read_csv(feature_filename, header=None)

            featureNum = feature_df.shape[0]

            feature_df_sort = feature_df.sort_values(by=3, ascending=False)

            topN_maxlen = feature_df_sort.shape[0]

            SFATest_filename = root + dataset + "/SFA_validate_" + data_type + ".csv"
            if os.path.getsize(SFATest_filename) > 0:
                SFATest_df = pd.read_csv(SFATest_filename, header=None)

            RISETest_filename = root + dataset + "/RISE_validate.csv"
            if os.path.getsize(RISETest_filename) > 0:
                RISETest_df = pd.read_csv(RISETest_filename, header=None)

            SAXTest_filename = root + dataset + "/SAX_validate_" + data_type + ".csv"
            if os.path.getsize(SAXTest_filename) > 0:
                SAXTest_df = pd.read_csv(SAXTest_filename, header=None)

            predict_labelCnt = []
            for i in range(testdata_size):
                tmp_dict = {}
                for datalabel in dataLabel:
                    tmp_dict[datalabel] = 0
                predict_labelCnt.append(tmp_dict)

            Predict_ls = []

            for topN in range(selectPerNum_label, topN_maxlen, selectPerNum_label):

                Predict_topN_ls = []
                Weight = []
                feature_topN_df = feature_df_sort.iloc[topN - selectPerNum_label: min(topN, topN_maxlen)]

                for feature_i, feature_row in feature_topN_df.iterrows():
                    feature = list(feature_row)
                    feature_id = feature[0]
                    feature_type = feature[1]
                    feature_label = feature[2]
                    feature_threshold = feature[4]
                    feature_infogain = feature[3]
                    feature_sign = feature[5]

                    Weight.append(feature_infogain)

                    test_tmp = None
                    feature_predictLs = []
                    if feature_type == "SFA":
                        test_tmp = list(SFATest_df[SFATest_df[0] == feature_id].values[0])[1:]
                        for test_value in test_tmp:
                            if test_value < feature_threshold:
                                feature_predictLs.append(feature_label)
                            else:
                                feature_predictLs.append(feature_label + 100000)
                    elif feature_type == "SAX":
                        test_tmp = list(SAXTest_df[SAXTest_df[0] == feature_id].values[0])[1:]
                        for test_value in test_tmp:
                            if test_value < feature_threshold:
                                feature_predictLs.append(feature_label)
                            else:
                                feature_predictLs.append(feature_label + 100000)
                    elif feature_type == "RISE":
                        test_tmp = list(RISETest_df[RISETest_df[0] == feature_id].values[0])[1:]
                        for test_value in test_tmp:
                            if feature_sign == 1:
                                if test_value >= feature_threshold:
                                    feature_predictLs.append(feature_label)
                                else:
                                    feature_predictLs.append(feature_label + 100000)
                            else:
                                if test_value < feature_threshold:
                                    feature_predictLs.append(feature_label)
                                else:
                                    feature_predictLs.append(feature_label + 100000)

                    Predict_topN_ls.append(feature_predictLs)

                Predict_topN_df = pd.DataFrame(Predict_topN_ls)

                Predict_topN_result_ls = []
                for idx, col in Predict_topN_df.iteritems():
                    predict_ls = col.values
                    predict_dict = predict_labelCnt[idx]
                    # print(predict_dict)
                    # print(predict_ls)

                    for predict_sfa_id, predict_val in enumerate(predict_ls):
                        # print(predict_val,SFAWeight[predict_sfa_id],end="||")
                        if change_label:
                            if predict_val > 50000:
                                predict_dict[predict_val - 100000] = predict_dict[predict_val - 100000] - Weight[
                                    predict_sfa_id]
                            else:
                                predict_dict[predict_val] = predict_dict[predict_val] + Weight[predict_sfa_id]
                        else:
                            if predict_val > 50000:
                                continue
                            else:
                                predict_dict[predict_val] = predict_dict[predict_val] + Weight[predict_sfa_id]
                    predict_labelCnt[idx] = predict_dict

                    predict_val_cnt = -1000000
                    predict_label = 0
                    flag = False
                    # print(predict_dict)
                    for predict_val in predict_dict:
                        if predict_dict[predict_val] > predict_val_cnt:
                            predict_val_cnt = predict_dict[predict_val]
                            predict_label = predict_val
                            flag = False
                        elif predict_dict[predict_val] == predict_val_cnt:
                            flag = True
                    if flag is True:
                        predict_label = 1000000
                    Predict_topN_result_ls.append(predict_label)

                trueCnt = 0
                for i in range(0, len(Predict_topN_result_ls)):
                    if test_label[i] == Predict_topN_result_ls[i]:
                        trueCnt += 1
                acc = trueCnt / len(test_label)

                topN_acc.append(acc)

            # print("\n")
            # print(topN_acc)
            best_topN_id = 0
            best_acc = 0
            for i in range(0, len(topN_acc)):
                if choose_compare == 1:
                    if topN_acc[i] > best_acc:
                        best_acc = topN_acc[i]
                        best_topN_id = i
                elif choose_compare == 2:
                    if topN_acc[i] >= best_acc:
                        best_acc = topN_acc[i]
                        best_topN_id = i
                elif choose_compare == 3:
                    if topN_acc[i] > best_acc:
                        best_acc = topN_acc[i]
                        best_topN_id = i
                    elif topN_acc[i] == best_acc and (
                            i * selectPerNum_label + selectPerNum_label) <= pp * featureNum:
                        best_acc = topN_acc[i]
                        best_topN_id = i
            best_top_n = best_topN_id * selectPerNum_label + selectPerNum_label
            print("Extract Feature Num: ", featureNum)
            print("best_topN: ", best_top_n)
            print("best acc:", best_acc)

            result_acc_path = root + dataset + "/topN_acc.txt"
            with open(result_acc_path, 'w') as f:
                f.write("Extract Feature Num: " + str(featureNum) + "\n")
                f.write("best_topN: " + str(best_top_n) + "\n")
                f.write("best acc:" + str(best_acc) + "\n")

            best_topN_ls.append(best_top_n)

    if update_topN is True:
        for datasetId, dataset in enumerate(datasets):
            topN_path = root + dataset + "/sfa_cluster"
            if os.path.exists(topN_path):
                shutil.rmtree(topN_path)
                os.mkdir(topN_path)

            if not os.path.exists(topN_path):
                os.mkdir(topN_path)

            featureNum = best_topN_ls[datasetId]

            ALLfeature_filename = root + dataset + "/ALLfeature_" + data_type + ".csv"
            ALLfeature_df = pd.read_csv(ALLfeature_filename, header=None)

            SAX_feature_filename = root + dataset + "/SAXfeature.csv"
            SAX_featureDf = None
            SAX_validateDf = None
            SAX_testDf = None
            if os.path.getsize(SAX_feature_filename) > 0:
                SAX_featureDf = pd.read_csv(SAX_feature_filename, header=None)

                SAX_validate_filename = root + dataset + "/SAX_validate_" + data_type + ".csv"
                SAX_validateDf = pd.read_csv(SAX_validate_filename, header=None)

                SAX_test_filename = root + dataset + "/SAX_test_" + data_type + ".csv"
                SAX_testDf = pd.read_csv(SAX_test_filename, header=None)

            SFA_feature_filename = root + dataset + "/SFAfeature_" + data_type + ".csv"
            SFA_featureDf = None
            SFA_validateDf = None
            SFA_testDf = None
            if os.path.getsize(SFA_feature_filename) > 0:
                SFA_featureDf = pd.read_csv(SFA_feature_filename, header=None)

                SFA_validate_filename = root + dataset + "/SFA_validate_" + data_type + ".csv"
                SFA_validateDf = pd.read_csv(SFA_validate_filename, header=None)

                SFA_test_filename = root + dataset + "/SFA_test_" + data_type + ".csv"
                SFA_testDf = pd.read_csv(SFA_test_filename, header=None)

            RISE_feature_filename = root + dataset + "/RISEfeature.csv"
            RISE_featureDf = None
            RISE_validateDf = None
            RISE_testDf = None
            if os.path.getsize(RISE_feature_filename) > 0:
                RISE_featureDf = pd.read_csv(RISE_feature_filename, header=None)

                RISE_validate_filename = root + dataset + "/RISE_validate.csv"
                RISE_validateDf = pd.read_csv(RISE_validate_filename, header=None)

                RISE_test_filename = root + dataset + "/RISE_test.csv"
                RISE_testDf = pd.read_csv(RISE_test_filename, header=None)

            ALLfeatureNum = ALLfeature_df.shape[0]
            ALLfeature_df_sort = ALLfeature_df.sort_values(by=3, ascending=False)

            ALLfeature_topN_df = pd.DataFrame()
            SFAfeature_topN_df = pd.DataFrame()
            RISEfeature_topN_df = pd.DataFrame()
            SAXfeature_topN_df = pd.DataFrame()

            SFAvalidate_topN_df = pd.DataFrame()
            RISEvalidate_topN_df = pd.DataFrame()
            SAXvalidate_topN_df = pd.DataFrame()

            SFAtest_topN_df = pd.DataFrame()
            RISEtest_topN_df = pd.DataFrame()
            SAXtest_topN_df = pd.DataFrame()

            ALLfeature_label_topN_df = ALLfeature_df_sort.iloc[0: featureNum]
            ALLfeature_topN_df = pd.concat([ALLfeature_topN_df, ALLfeature_label_topN_df])

            for idx, rows in ALLfeature_label_topN_df.iterrows():
                feature_tmp = list(rows)
                featureId = feature_tmp[0]
                featureType = feature_tmp[1]

                if featureType == "SFA":
                    SFAfeature_topN_df = pd.concat(
                        [SFAfeature_topN_df, SFA_featureDf[SFA_featureDf[0] == featureId]])
                    SFAvalidate_topN_df = pd.concat(
                        [SFAvalidate_topN_df, SFA_validateDf[SFA_validateDf[0] == featureId]])
                    SFAtest_topN_df = pd.concat([SFAtest_topN_df, SFA_testDf[SFA_testDf[0] == featureId]])

                elif featureType == "SAX":
                    SAXfeature_topN_df = pd.concat(
                        [SAXfeature_topN_df, SAX_featureDf[SAX_featureDf[0] == featureId]])
                    SAXvalidate_topN_df = pd.concat(
                        [SAXvalidate_topN_df, SAX_validateDf[SAX_validateDf[0] == featureId]])
                    SAXtest_topN_df = pd.concat([SAXtest_topN_df, SAX_testDf[SAX_testDf[0] == featureId]])

                elif featureType == "RISE":
                    RISEfeature_topN_df = pd.concat(
                        [RISEfeature_topN_df, RISE_featureDf[RISE_featureDf[0] == featureId]])
                    RISEvalidate_topN_df = pd.concat(
                        [RISEvalidate_topN_df, RISE_validateDf[RISE_validateDf[0] == featureId]])
                    RISEtest_topN_df = pd.concat([RISEtest_topN_df, RISE_testDf[RISE_testDf[0] == featureId]])

            featureNum_result_file = topN_path + "/featureNum_topN.txt"
            with open(featureNum_result_file, "w") as f:
                f.write("SFAfeature个数：" + str(SFAfeature_topN_df.shape[0]) + "\n")
                f.write("SAXfeature个数：" + str(SAXfeature_topN_df.shape[0]) + "\n")
                f.write("RISEfeature个数：" + str(RISEfeature_topN_df.shape[0]) + "\n")
                f.close()

            result_file = topN_path + "/feature_topN.csv"
            ALLfeature_topN_df.to_csv(result_file, header=None, index=None)

            if os.path.getsize(SAX_feature_filename) > 0:
                sax_result_file = topN_path + "/SAXfeature_topN.csv"
                SAXfeature_topN_df.to_csv(sax_result_file, header=None, index=None)

                sax_validate_file = topN_path + "/SAX_validate_topN.csv"
                SAXvalidate_topN_df.to_csv(sax_validate_file, header=None, index=None)

                sax_test_file = topN_path + "/SAX_test_topN.csv"
                SAXtest_topN_df.to_csv(sax_test_file, header=None, index=None)

            if os.path.getsize(SFA_feature_filename) > 0:
                sfa_result_file = topN_path + "/SFAfeature_topN.csv"
                SFAfeature_topN_df.to_csv(sfa_result_file, header=None, index=None)

                sfa_validate_file = topN_path + "/SFA_validate_topN.csv"
                SFAvalidate_topN_df.to_csv(sfa_validate_file, header=None, index=None)

                sfa_test_file = topN_path + "/SFA_test_topN.csv"
                SFAtest_topN_df.to_csv(sfa_test_file, header=None, index=None)

            if os.path.getsize(RISE_feature_filename) > 0:
                rise_result_file = topN_path + "/RISEfeature_topN.csv"
                RISEfeature_topN_df.to_csv(rise_result_file, header=None, index=None)

                rise_validate_file = topN_path + "/RISE_validate_topN.csv"
                RISEvalidate_topN_df.to_csv(rise_validate_file, header=None, index=None)

                rise_test_file = topN_path + "/RISE_test_topN.csv"
                RISEtest_topN_df.to_csv(rise_test_file, header=None, index=None)