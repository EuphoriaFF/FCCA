package util;

import util.pair.Pair;

import java.util.ArrayList;
import java.util.HashSet;

public class FeatureUtil {

    public static double getEntropy(ArrayList<Integer> label_ls, int label){
        int self_cnt = 0;
        int other_cnt = 0;

        for (Integer x : label_ls) {
            if(x==label){
                self_cnt++;
            }else{
                other_cnt++;
            }
        }

        if(self_cnt==0 || other_cnt==0){
            return 0;
        }

        double self_p = (self_cnt*1.0) / label_ls.size();
        double other_p = (other_cnt*1.0) / label_ls.size();
        double entropy = -self_p * Math.log(self_p) - other_p*Math.log(other_p);

        return entropy;
    }

    public static Pair<Double, Integer> getInfoGain(ArrayList<Integer> sfaList, ArrayList<Integer> test_label, int label){
        //划分前的熵
        double entropy_origin = getEntropy(test_label, label);

        //得到分割点
        HashSet<Integer> sfaList_unique = new HashSet<>(sfaList);

        double best_sp_infogain = 0;
        int best_sp_val = 0;
        int n = sfaList.size();
        for (int sp : sfaList_unique) {

            ArrayList<Integer> sfaLabel1 = new ArrayList<>();
            ArrayList<Integer> sfaLabel2 = new ArrayList<>();

            for(int i=0;i<n;i++){
                int ss = sfaList.get(i);
                if(ss>=sp) sfaLabel1.add(test_label.get(i));
                else sfaLabel2.add(test_label.get(i));
            }

            double entropy1 = getEntropy(sfaLabel1, label);
            double entropy2 = getEntropy(sfaLabel2, label);

            double sp_infogain = entropy_origin - (sfaLabel1.size()*1.0/test_label.size())*entropy1 - (sfaLabel2.size()*1.0/test_label.size())*entropy2;

            if(sp_infogain>best_sp_infogain){
                best_sp_infogain = sp_infogain;
                best_sp_val = sp;
            }

        }

        return new Pair<Double, Integer>(best_sp_infogain, best_sp_val);
    }

    public static Pair<Double, Double> getInfoGain_double(ArrayList<Double> sfaList, ArrayList<Integer> test_label, int label){
        //划分前的熵
        double entropy_origin = getEntropy(test_label, label);

        //得到分割点
        HashSet<Double> sfaList_unique = new HashSet<>(sfaList);

        double best_sp_infogain = 0;
        double best_sp_val = 0;
        int n = sfaList.size();
        for (double sp : sfaList_unique) {

            ArrayList<Integer> sfaLabel1 = new ArrayList<>();
            ArrayList<Integer> sfaLabel2 = new ArrayList<>();

            for(int i=0;i<n;i++){
                double ss = sfaList.get(i);
                if(ss>=sp) sfaLabel1.add(test_label.get(i));
                else sfaLabel2.add(test_label.get(i));
            }

            double entropy1 = getEntropy(sfaLabel1, label);
            double entropy2 = getEntropy(sfaLabel2, label);

            double sp_infogain = entropy_origin - (sfaLabel1.size()*1.0/test_label.size())*entropy1 - (sfaLabel2.size()*1.0/test_label.size())*entropy2;

            if(sp_infogain>best_sp_infogain){
                best_sp_infogain = sp_infogain;
                best_sp_val = sp;
            }
        }

        return new Pair<Double, Double>(best_sp_infogain, best_sp_val);
    }



}
