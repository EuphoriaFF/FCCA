package dev;

import core.AppContext;
import datasets.SaxDataset;
import datasets.SaxDataset.SAXParams;
import datasets.TSDataset;
import datasets.TimeSeries;
import transforms.SAX;

import java.util.*;

public class SAXTransformerContainer implements TransformContainer {

    //需要随机选择的参数
    int alphabetSize = 4; //字母表大小（这里固定）

    int minWindowLength = 10; //窗口长度
    int maxWindowLength = 0; //if 0 then assume maxWindowLength == length of series
    int windowLengthStepSize = 1;

    int minPaaSize = 6; //PAA分段个数
    int maxPaaSize = 20;
    int paaSizeStepsize = 2;

    double nThreshold = 0.001; //  判断PAA时是否要标准化数据：<threshold -> 不对这个序列数据标准化

    int num_transforms; //最多有多少种转换

    public HashMap<String, SaxDataset> sax_datasets;
    public List<SAXParams> sax_params;
    public HashMap<String, HashMap<Integer, HashMap<String, Double>>> sax_tfidf;//每个sax参数下对应的tfidf计算列表

    public HashMap<String, ArrayList<ArrayList<String>>> datasets_to_sax;

    public static final char[] ALPHABET = new char[]{'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};


    public SAXTransformerContainer(int num_transforms) {
        sax_datasets = new HashMap<>(num_transforms);
        sax_params = new ArrayList<>(num_transforms);
        sax_tfidf = new HashMap<>(num_transforms);
        datasets_to_sax = new HashMap<>(num_transforms);
        this.num_transforms = num_transforms;
    }

    private ArrayList<Integer> selectRandomParams(TSDataset train){
        if (maxWindowLength == 0 || maxWindowLength > train.length()) {
            maxWindowLength = train.length();
        }

        List<SAXParams> all_params = new ArrayList<>();
        for (int w = minWindowLength; w < maxWindowLength; w+=windowLengthStepSize) {
            for(int p = minPaaSize; p<maxPaaSize;p+=paaSizeStepsize){
                all_params.add(new SaxDataset.SAXParams(alphabetSize, w, p, nThreshold));
            }
        }

        ArrayList<Integer> indices = new ArrayList<>(all_params.size());
        for (int i = 0; i < all_params.size(); i++) {
            indices.add(i);
        }

        Collections.shuffle(indices);

        if (num_transforms > all_params.size()) {
            num_transforms = all_params.size();
            if (AppContext.verbosity > 1) {
                System.out.println("INFO: sax_num_transformation has been updated to: " + num_transforms + " (#all possible params with given settings ("+all_params.size()+") < the given num_transformations)");
            }
        }

        for (int i = 0; i < num_transforms; i++){
            SaxDataset.SAXParams params = all_params.get(indices.get(i));
            sax_params.add(new SaxDataset.SAXParams(params.alphabet_size, params.window_len, params.Paa_size, nThreshold));
        }

        all_params = null;

        return indices;

    }

    //在所有随机生成的参数组合下，得到sax转换后的数据
    @Override
    public void fit(TSDataset train) {
        TimeSeries[] samples = train._get_internal_list().toArray(new TimeSeries[] {});

        selectRandomParams(train);

        for (int paramsId = 0; paramsId < sax_params.size(); paramsId++) {
            SaxDataset.SAXParams params = sax_params.get(paramsId);

            SaxDataset saxDataset1 = transform_data_using_sax_no_tfidf_withSaxLs(params, samples);
            SAX.BagOfPattern[] sax_transformed_data = saxDataset1.sax_transformed_data;

            sax_datasets.put(params.toString(), saxDataset1);

        }
    }

    @Override
    public TSDataset transform(TSDataset test) {
//        System.out.println("computing test transformations: " + num_transforms);
//        TimeSeries[] samples = test._get_internal_list().toArray(new TimeSeries[] {});
//
//        for (int paramsId = 0; paramsId < sax_params.size(); paramsId++) {
//            SaxDataset.SAXParams params = sax_params.get(paramsId);
//
//            // 得到转换数据
//            Classifier.Pair<ArrayList<HashMap<String, Double>>, ArrayList<ArrayList<String>>> res_tmp = get_dataset_freqHistAndsaxls(params, samples);
////            ArrayList<HashMap<String, Double>> dataset_bag = get_dataset_freqHist(params, samples);
//            ArrayList<HashMap<String, Double>> dataset_bag = res_tmp.key;
//            ArrayList<ArrayList<String>> dataset_saxword_ls = res_tmp.value;
//            HashMap<Integer, HashMap<String, Double>> dictionary_tfidf = sax_tfidf.get(params.toString());
//            SaxDataset saxDataset = transform_data_using_sax_with_tfidf(dataset_bag, dictionary_tfidf, samples, params);
//
//            sax_datasets.put(params.toString(), saxDataset);
//
//        }

        return null;
    }

    public double mean(double[] series) {
        double res = 0.0D;
        int count = 0;
        double[] var5 = series;
        int var6 = series.length;

        for(int var7 = 0; var7 < var6; ++var7) {
            double tp = var5[var7];
            res += tp;
            ++count;
        }

        return count > 0 ? res / Integer.valueOf(count).doubleValue() : 0.0D / 0.0;
    }

    public double stDev(double[] series) {
        double num0 = 0.0D;
        double sum = 0.0D;
        int count = 0;
        double[] var7 = series;
        int var8 = series.length;

        for(int var9 = 0; var9 < var8; ++var9) {
            double tp = var7[var9];
            num0 += tp * tp;
            sum += tp;
            ++count;
        }

        double len = Integer.valueOf(count).doubleValue();
        return Math.sqrt((len * num0 - sum * sum) / (len * (len - 1.0D)));
    }

    public double[] znorm(double[] series, double normalizationThreshold) {
        double[] res = new double[series.length];
        double mean = this.mean(series);
        double sd = this.stDev(series);
        if (sd < normalizationThreshold) {
            return res;
        } else {
            for(int i = 0; i < res.length; ++i) {
                res[i] = (series[i] - mean) / sd;
            }

            return res;
        }
    }

    public double[] paa(double[] ts, int paaSize){
        int len = ts.length;
        if (len <= paaSize) {
            return Arrays.copyOf(ts, ts.length);
        } else {
            double[] paa = new double[paaSize];
            double pointsPerSegment = (double)len / (double)paaSize;
            double[] breaks = new double[paaSize + 1];

            int i;
            for(i = 0; i < paaSize + 1; ++i) {
                breaks[i] = (double)i * pointsPerSegment;
            }

            for(i = 0; i < paaSize; ++i) {
                double segStart = breaks[i];
                double segEnd = breaks[i + 1];
                double fractionStart = Math.ceil(segStart) - segStart;
                double fractionEnd = segEnd - Math.floor(segEnd);
                int fullStart = Double.valueOf(Math.floor(segStart)).intValue();
                int fullEnd = Double.valueOf(Math.ceil(segEnd)).intValue();
                double[] segment = Arrays.copyOfRange(ts, fullStart, fullEnd);
                if (fractionStart > 0.0D) {
                    segment[0] *= fractionStart;
                }

                if (fractionEnd > 0.0D) {
                    segment[segment.length - 1] *= fractionEnd;
                }

                double elementsSum = 0.0D;
                double[] var22 = segment;
                int var23 = segment.length;

                for(int var24 = 0; var24 < var23; ++var24) {
                    double e = var22[var24];
                    elementsSum += e;
                }

                paa[i] = elementsSum / pointsPerSegment;
            }

            return paa;
        }
    }

    public char num2char(double value, double[] cuts) {
        int count;
        for(count = 0; count < cuts.length && cuts[count] <= value; ++count) {
            ;
        }

        return ALPHABET[count];
    }

    public char[] ts2String(double[] vals, double[] cuts) {
        char[] res = new char[vals.length];

        for(int i = 0; i < vals.length; ++i) {
            res[i] = this.num2char(vals[i], cuts);
        }

        return res;
    }

    // 找到一个saxword在一个序列中的起始位置（用于后续观测sax形状）
    public int findSaxwordSt(SAXParams params, TimeSeries sample, String example){

        int window_len = params.window_len;
        int ts_len = sample.getLength();
        double[] ts = sample.getData();

        char[] previousString = null;

        NormalAlphabet na = new NormalAlphabet();

        for(int i=0;i<= ts_len-window_len; i++){

            double[] subSection = Arrays.copyOfRange(ts, i, i + window_len);

            //z normalize
            subSection = znorm(subSection, params.nThreshold);

            double[] paa = paa(subSection, params.Paa_size);

            char[] currentString = ts2String(paa, na.getCuts(params.alphabet_size));

            if (null != previousString){
                if (Arrays.equals(previousString, currentString)){
                    continue;
                }
            }

            previousString = currentString;
            String current_string = String.valueOf(currentString);

            if(current_string.equals(example)){
                return i;
            }

        }

        return -1;
    }


    public ArrayList<String> ts_to_saxwordls(SAXParams params, TimeSeries sample){
        int window_len = params.window_len;
        int ts_len = sample.getLength();
        double[] ts = sample.getData();

        ArrayList<String> sax_ls = new ArrayList<>();
        NormalAlphabet na = new NormalAlphabet();

        for(int i=0;i<= ts_len-window_len; i++){

            double[] subSection = Arrays.copyOfRange(ts, i, i + window_len);

            //z normalize
            subSection = znorm(subSection, params.nThreshold);

            double[] paa = paa(subSection, params.Paa_size);

            char[] currentString = ts2String(paa, na.getCuts(params.alphabet_size));

            String current_string = String.valueOf(currentString);
            sax_ls.add(current_string);

        }

        return sax_ls;
    }

    public Classifier.Pair<HashMap<String, Double>, ArrayList<String>> ts_to_bagAndwordls(SAXParams params, TimeSeries sample){
        int window_len = params.window_len;
        int ts_len = sample.getLength();
        double[] ts = sample.getData();

        HashMap<String, Double> ts_bag = new HashMap<>();
        ArrayList<String> sax_ls = new ArrayList<>();
        char[] previousString = null;

        NormalAlphabet na = new NormalAlphabet();

        for(int i=0;i<= ts_len-window_len; i++){

            double[] subSection = Arrays.copyOfRange(ts, i, i + window_len);

            //z normalize
            subSection = znorm(subSection, params.nThreshold);

            double[] paa = paa(subSection, params.Paa_size);

            char[] currentString = ts2String(paa, na.getCuts(params.alphabet_size));

            String current_string = String.valueOf(currentString);
            sax_ls.add(current_string);

            if (null != previousString){
                if (Arrays.equals(previousString, currentString)){
                    continue;
                }
            }

            previousString = currentString;

//            double saxword_cnt = ts_bag.getOrDefault(current_string, 0.0)+1.0;
//
//            ts_bag.put(current_string, saxword_cnt);

        }

//        System.out.println(ts_len+","+window_len);
//        System.out.println(ts_bag.keySet().size());
//        System.out.println(sax_ls.size());

        Classifier.Pair<HashMap<String, Double>, ArrayList<String>> result = Classifier.Pair.create(ts_bag, sax_ls);

        return result;
    }
    
    //用于数据集中所有数据的直方图表示
//    public ArrayList<HashMap<String, Double>> get_dataset_freqHist(SAXParams params, TimeSeries[] samples){
//        ArrayList<HashMap<String, Double>> dataset_bag = new ArrayList<>(); //每个数据对应的直方图
//
//        for (TimeSeries sample : samples) {
//            HashMap<String, Double> ts_bag = ts_to_bag(params, sample); // <sax_word, 出现次数>
//            dataset_bag.add(ts_bag);
//        }
//        return dataset_bag;
//    }

    //用于数据集中所有数据的直方图表示和sax word转换
//    public Classifier.Pair<ArrayList<HashMap<String, Double>>, ArrayList<ArrayList<String>>> get_dataset_freqHistAndsaxls(SAXParams params, TimeSeries[] samples){
//        ArrayList<HashMap<String, Double>> dataset_bag = new ArrayList<>(); //每个数据对应的直方图
//        ArrayList<ArrayList<String>> dataset_saxwordLs = new ArrayList<>();
//
//        for (TimeSeries sample : samples) {
//            Classifier.Pair<HashMap<String, Double>, ArrayList<String>> ts_to_bagAndwordls = ts_to_bagAndwordls(params, sample);
//            HashMap<String, Double> ts_bag = ts_to_bagAndwordls.key; // <sax_word, 出现次数>
//            dataset_bag.add(ts_bag);
//            dataset_saxwordLs.add(ts_to_bagAndwordls.value);
//        }
//
//        Classifier.Pair<ArrayList<HashMap<String, Double>>, ArrayList<ArrayList<String>>> res = Classifier.Pair.create(dataset_bag, dataset_saxwordLs);
//        return res;
//    }

//    public SaxDataset transform_data_using_sax_no_tfidf(SAXParams params, TimeSeries[] samples){
//
//        SAX.BagOfPattern[] sax_bag = new SAX.BagOfPattern[samples.length];
//
//        for (int i = 0; i < samples.length; i++) {
//            TimeSeries sample = samples[i];
//            HashMap<String, Double> ts_bag = ts_to_bag(params, sample); // <sax_word, 出现次数>
//
//            Integer tsLabel = sample.getLabel();
//            sax_bag[i] = new SAX.BagOfPattern(ts_bag, tsLabel, sample, i);
//        }
//
//
//        return new SaxDataset(sax_bag, params);
//    }

    public SaxDataset transform_data_using_sax_no_tfidf_withSaxLs(SAXParams params, TimeSeries[] samples){

        SAX.BagOfPattern[] sax_bag = new SAX.BagOfPattern[samples.length];

        ArrayList<ArrayList<String>> dataset_saxwordLs = new ArrayList<>();
        int sum = 0;

        for (int i = 0; i < samples.length; i++) {
            TimeSeries sample = samples[i];

            Classifier.Pair<HashMap<String, Double>, ArrayList<String>> ts_to_bagAndwordls = ts_to_bagAndwordls(params, sample);
            HashMap<String, Double> ts_bag = ts_to_bagAndwordls.key;

            Integer tsLabel = sample.getLabel();
            sax_bag[i] = new SAX.BagOfPattern(ts_bag, tsLabel, sample, i, ts_to_bagAndwordls.value);
            dataset_saxwordLs.add(ts_to_bagAndwordls.value);

        }
        datasets_to_sax.put(params.toString(), dataset_saxwordLs);

        return new SaxDataset(sax_bag, params);
    }
    
//    public HashMap<Integer, HashMap<String, Double>> get_dictionary_tfidf(TimeSeries[] samples, ArrayList<HashMap<String, Double>> dataset_bag){
//        HashMap<Integer, HashMap<String, Double>> dictionary_bag=new HashMap<>(); //每个类别下的单词直方图
//
//        HashMap<Integer, HashMap<String, Double>> dictionary_tfidf = new HashMap<>();
//
//        //1. 得到每个类别下的直方图统计（用于计算tf：每个单词在每个文件(类别)中出现的频率）
//        for (int i = 0; i < samples.length; i++) {
//            TimeSeries sample = samples[i];
//            int ts_label = sample.getLabel();
//            HashMap<String, Double> ts_bag = dataset_bag.get(i);
//
//            if(dictionary_bag.containsKey(ts_label)){
//                HashMap<String, Double> label_bag = dictionary_bag.get(ts_label);
//
//                for (String s : ts_bag.keySet()) {
//                    if(label_bag.containsKey(s)){
//                        label_bag.put(s,label_bag.get(s)+ts_bag.get(s));
//                    }else{
//                        label_bag.put(s, ts_bag.get(s));
//                    }
//                }
//                dictionary_bag.put(ts_label, label_bag);
//            }else{
//                dictionary_bag.put(ts_label, ts_bag);
//            }
//        }
//
//        //2. 得到tf_idf权重
//        //2.1 得到每个单词出现在几个类别中（df）
//        HashMap<String, Integer> words_inDocumentsCnt = new HashMap<>();
//        for (Integer dictionary_bag_class : dictionary_bag.keySet()) {
//            dictionary_tfidf.put(dictionary_bag_class, new HashMap<>());
//            HashMap<String, Double> bag_hist = dictionary_bag.get(dictionary_bag_class);
//
//            for (String bag_sax_word : bag_hist.keySet()) {
//                if(words_inDocumentsCnt.containsKey(bag_sax_word)){
//                    words_inDocumentsCnt.put(bag_sax_word, words_inDocumentsCnt.get(bag_sax_word)+1);
//                }else{
//                    words_inDocumentsCnt.put(bag_sax_word, 1);
//                }
//            }
//        }
//        //2.2 总文件数
//        int totalDocs = dictionary_bag.size();
//
//        //遍历每一个文件
//        for (Integer dictionary_bag_class : dictionary_bag.keySet()){
//            HashMap<String, Double> bag_hist = dictionary_bag.get(dictionary_bag_class);
//
//            for (String sax_word : words_inDocumentsCnt.keySet()) {
//                double tf_idf = 0;
//
//                if(bag_hist.containsKey(sax_word) && totalDocs!=words_inDocumentsCnt.get(sax_word)){
//                    double wordInBagfrequency = bag_hist.get(sax_word);
//
//                    double tfValue = Math.log(1.0+wordInBagfrequency);
//
//                    double idfValue = Math.log(totalDocs*1.0 / words_inDocumentsCnt.get(sax_word));
//
//                    tf_idf = tfValue*idfValue;
//                }
//
//                dictionary_tfidf.get(dictionary_bag_class).put(sax_word, tf_idf);
//
//            }
//        }
//
//        return dictionary_tfidf;
//    }

//    public SaxDataset transform_data_using_sax_with_tfidf(ArrayList<HashMap<String, Double>> dataset_bag, HashMap<Integer, HashMap<String, Double>> dictionary_tfidf, TimeSeries[] samples, SAXParams params){
//
//        //得到最终转换数据
//        SAX.BagOfPattern[] sax_bag = new SAX.BagOfPattern[dataset_bag.size()];
//        for (int i = 0; i < dataset_bag.size(); i++) {
//            HashMap<String, Double> single_dataset_bag = dataset_bag.get(i); //时序数据转换得到的 频率
//            TimeSeries sample = samples[i];
//            Integer tsLabel = sample.getLabel();
//
//            HashMap<String, Double> singleClass_tfidf = dictionary_tfidf.get(tsLabel); //该数据label下的tfidf列表
//
//            HashMap<String, Double> ts_tfidf_bag = new HashMap<>(); //时序数据最终转换后的结果
//            for (String s : single_dataset_bag.keySet()) {
//                Double tfidf_val = 0.0;
//                double freq_val = single_dataset_bag.get(s);
//
//                if(singleClass_tfidf.containsKey(s)){
//                    tfidf_val = singleClass_tfidf.get(s);
//                }
//
//                ts_tfidf_bag.put(s, freq_val*tfidf_val);
////                ts_tfidf_bag.put(s, freq_val*1.0);
//            }
//            sax_bag[i] = new SAX.BagOfPattern(ts_tfidf_bag, tsLabel, sample, i);
//
//        }
//
//        return new SaxDataset(sax_bag, params);
//
//    }

}
