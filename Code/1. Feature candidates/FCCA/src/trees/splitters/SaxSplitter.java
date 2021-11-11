package trees.splitters;

import core.AppContext;
import datasets.SaxDataset;
import datasets.SaxDataset.SAXParams;
import datasets.TSDataset;
import datasets.TimeSeries;
import dev.SAXTransformerContainer;
import dev.TransformContainer;
import feature.FeatureSAX;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import transforms.SAX;
import transforms.SAX.BagOfPattern;
import trees.ProximityTree;
import util.FeatureUtil;
import util.pair.Pair;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

public class SaxSplitter implements NodeSplitter  {

    public ProximityTree.Node node;

    public TIntObjectHashMap<BagOfPattern> bopPerClass;
    TIntObjectMap<TSDataset> best_splits = null; //每个类别下对应的数据集

    public TIntObjectHashMap<Pair<FeatureSAX, ArrayList<Double>>> bestSAX_PerClass;

    public SAXParams saxParams;

    double train_node_time = 0.0;

    public SaxSplitter(ProximityTree.Node node) {
        this.node = node;
        this.node.tree.stats.sax_count++;
    }

    @Override
    public TIntObjectMap<TSDataset> train(TSDataset data, int[] indices) throws Exception {

        long startTime = System.nanoTime();
        boolean sax_ok = get_best_splitter_by_distance(data, indices);
        if(!sax_ok) return null;

        this.node.tree.stats.sax_splitter_train_time += (System.nanoTime() - startTime);

        return split(data, indices);
    }


    protected double SaxDistance_map(HashMap<String, Double> queryBag, HashMap<String, Double> exampleBag) {

        double distance = 0;

        for (String s : queryBag.keySet()) {
            double tmp = 0;
            if(exampleBag.containsKey(s)){
                tmp = queryBag.get(s)-exampleBag.get(s);
            }else{
                tmp = queryBag.get(s);
            }
            distance += tmp*tmp;
        }
        return distance;
    }

//    List<Integer> closest_nodes = new ArrayList<Integer>();

    public boolean check_sax_sp_valid_with_dis(ArrayList<Double> ls, double threshold, ArrayList<Integer> test_label, int seriesLabel){
        int large_th_cnt = 0, less_th_cnt = 0;
        for (int i = 0; i < ls.size(); i++) {
            if(ls.get(i)>=threshold && test_label.get(i)==seriesLabel){
                large_th_cnt++;
            }
            if(ls.get(i)<threshold && test_label.get(i)==seriesLabel){
                less_th_cnt++;
            }
        }
        if(large_th_cnt>less_th_cnt) return false;
        return true;
    }

    public Pair<FeatureSAX, ArrayList<Double>> getBestSax_from_example(BagOfPattern example_hist, ArrayList<BagOfPattern> node_sax_dataset, ArrayList<Integer> node_sax_dataset_label){

        int seriesId = example_hist.getSeriesId();
        int seriesLabel = example_hist.getLabel();
        ArrayList<String> example_sax_ls = example_hist.sax_ls;

        //选出信息增益最大的sax
        String best_saxword_string = null;
        int best_saxword_id = 0;
        double infogain_max = -1;
        double best_sp_threshold = 0;
        ArrayList<Double> best_sax_dis_ls = null;

        HashMap<String, FeatureSAX> saxword_to_feature = new HashMap<>();
        ArrayList<FeatureSAX> best_sax_ls = new ArrayList<>();

        int example_sax_size = example_sax_ls.size();
        String pre_sax_word = null;
        for (int sax_word_id = 0; sax_word_id < example_sax_size; sax_word_id++) {

            String example_saxword_string = example_sax_ls.get(sax_word_id);
            if(example_saxword_string.equals(pre_sax_word)){
                continue;
            }
            if(saxword_to_feature.containsKey(example_saxword_string)){
                //只修改起始位置
                FeatureSAX best_sax_tmp = saxword_to_feature.get(example_saxword_string);
                if(best_sax_tmp==null){
                    continue;
                }
                best_sax_ls.add(new FeatureSAX(-1, "SAX", seriesLabel, best_sax_tmp.infogain, best_sax_tmp.threshold, 0,best_sax_tmp.sax_word, best_sax_tmp.origin_ts_Id, saxParams, sax_word_id));
                continue;
            }

            int example_saxword_len = example_saxword_string.length();
            pre_sax_word = example_saxword_string;

            //该sax_word与每个数据之间的距离
            ArrayList<Double> sax_dis_ls = new ArrayList<>();
            for (BagOfPattern sax_dataset_tmp : node_sax_dataset) {

                ArrayList<String> ts_saxword_ls = sax_dataset_tmp.sax_ls;
                HashSet<String> ts_saxword_set = new HashSet<>(ts_saxword_ls);
                double dis_min = Double.MAX_VALUE;

                for (String saxword_tmp : ts_saxword_set) {

                    double dis_tmp = 0;
                    for (int i = 0; i < example_saxword_len; i++) {
                        dis_tmp += Math.abs(example_saxword_string.charAt(i)-saxword_tmp.charAt(i));
                    }
                    if(dis_min>dis_tmp){
                        dis_min = dis_tmp;
                    }
                }
                sax_dis_ls.add(dis_min);
            }
            Pair<Double, Double> infoGain_tmp = FeatureUtil.getInfoGain_double(sax_dis_ls, node_sax_dataset_label, seriesLabel);

            if(infoGain_tmp.getKey()>=infogain_max){

                best_sp_threshold = infoGain_tmp.getValue();
                best_saxword_id = sax_word_id;
                best_saxword_string = example_saxword_string;
                best_sax_dis_ls = sax_dis_ls;

                if(!check_sax_sp_valid_with_dis(best_sax_dis_ls, best_sp_threshold, node_sax_dataset_label, seriesLabel)){
                    continue;
                }

                if(infoGain_tmp.getKey()>infogain_max){
                    best_sax_ls.clear();
                }

                infogain_max = infoGain_tmp.getKey();
                FeatureSAX best_sax_tmp = new FeatureSAX(-1, "SAX", seriesLabel, infogain_max, best_sp_threshold, 0,best_saxword_string, seriesId, saxParams, best_saxword_id);
                best_sax_ls.add(best_sax_tmp);
                saxword_to_feature.put(best_saxword_string, best_sax_tmp);

            }else{
                saxword_to_feature.put(example_saxword_string,null); // 表示这个word出现过，但是计算后对最佳SFA没有贡献
            }

        }

        int best_sax_ls_size = best_sax_ls.size();
        if(best_sax_ls_size==0){
            return null;
        }
        Random r = new Random();
        int choose_best_id = r.nextInt(best_sax_ls_size);

        FeatureSAX best_sax = best_sax_ls.get(choose_best_id);
        return new Pair<>(best_sax, best_sax_dis_ls);
    }

    public boolean get_best_splitter_by_distance(TSDataset data, int[] indices) throws Exception{

        long startTime = System.nanoTime();

        train_node_time = 0.0;

        //随机选取一组参数
        SAXTransformerContainer sax_transform = ((SAXTransformerContainer)this.node.tree.getForest().getTransforms().get("sax"));

        SplittableRandom rand = new SplittableRandom();
        int r = rand.nextInt(sax_transform.sax_params.size());
        saxParams = sax_transform.sax_params.get(r);

        // 得到该参数转换后的数据
        SaxDataset saxDataset = sax_transform.sax_datasets.get(saxParams.toString());
        BagOfPattern[] sax_transformed_data = saxDataset.sax_transformed_data;

        ArrayList<BagOfPattern> node_sax_dataset = new ArrayList<BagOfPattern>();
        for (int j = 0; j < indices.length; j++){
            node_sax_dataset.add(sax_transformed_data[indices[j]]);
        }

        ArrayList<Integer> node_sax_dataset_label = new ArrayList<>();
        for (BagOfPattern bag : node_sax_dataset) {
            node_sax_dataset_label.add(bag.getLabel());
        }

        //将该节点上的数据分类 <label, 该label下的数据的直方图的集合>
        TIntObjectMap<List<BagOfPattern>> sax_data_per_class = split_by_class(node_sax_dataset);

        long preTime = System.nanoTime()-startTime;

        //pick one random example per class
        bopPerClass = new TIntObjectHashMap<>();
        bestSAX_PerClass = new TIntObjectHashMap<>();

        boolean flag = true;

        for (int key : sax_data_per_class.keys()) {
            Pair<FeatureSAX, ArrayList<Double>> bestSax_from_example;
            BagOfPattern example;
            r = rand.nextInt(sax_data_per_class.get(key).size());
            example = sax_data_per_class.get(key).get(r);

            bestSax_from_example = getBestSax_from_example(example, node_sax_dataset, node_sax_dataset_label);
            if(bestSax_from_example == null){ //该splitter无效（最佳SFA都无法有效分开）
                return false;
            }

            bestSAX_PerClass.put(key, bestSax_from_example);
            bopPerClass.put(key, example);

            if(flag){
                int example_sfa_num = example.sax_ls.size();
                int data_num = node_sax_dataset.size();
                flag=false;

                int word_length = example.sax_ls.get(0).length();
                int wordset_m = node_sax_dataset.get(0).sax_ls.size();
                double time = (word_length/8.0)*(wordset_m/116.0)* 0.000089 * example_sfa_num * data_num * sax_data_per_class.keys().length;
                train_node_time = time;
            }

        }

        double d = Double.MAX_VALUE;
        TIntObjectMap<TSDataset> splits = new TIntObjectHashMap<TSDataset>();

        startTime = System.nanoTime();

        for (int j = 0; j < node_sax_dataset.size(); j++) {

            double maxInfogain = -1.0;
            int min_key = -1;
            double d_min = Double.MAX_VALUE;
            int d_min_key = -1;
            List<Integer> closest_nodes = new ArrayList<Integer>();

            for (int key : bestSAX_PerClass.keys()) {
                splits.putIfAbsent(key, new TSDataset());

                d = bestSAX_PerClass.get(key).getValue().get(j);
                FeatureSAX featureSAX = bestSAX_PerClass.get(key).getKey();

                if (d < featureSAX.threshold) {
                    if(featureSAX.infogain>maxInfogain){
                        maxInfogain = featureSAX.infogain;
                        closest_nodes.clear();
                        closest_nodes.add(key);
                    }else if(featureSAX.infogain==maxInfogain){
                        closest_nodes.add(key);
                    }
                }

                if(d-featureSAX.threshold<d_min){
                    d_min = d-featureSAX.threshold;
                    d_min_key = key;
                }

            }
            if(closest_nodes.size()==0){
                min_key = d_min_key;
            }else {
                r = ThreadLocalRandom.current().nextInt(closest_nodes.size());	//TODO may be use SplitRandom??
                min_key =  closest_nodes.get(r);
            }

            //put n to bin with min distance
            splits.get(min_key).add(node_sax_dataset.get(j).getSeries());
        }

        this.best_splits = splits;

        long afterTime = System.nanoTime() - startTime;

        train_node_time += (afterTime/AppContext.NANOSECtoMILISEC);

        return true;

    }

    public TIntObjectMap<List<BagOfPattern>> split_by_class(List<BagOfPattern> transformed_data) {
        TIntObjectMap<List<BagOfPattern>> split =  new TIntObjectHashMap<List<BagOfPattern>>();
        Integer label;
        List<BagOfPattern> class_set = null;

        for (int i = 0; i < transformed_data.size(); i++) {
            label = transformed_data.get(i).getLabel();
            if (! split.containsKey(label)) {
                class_set = new ArrayList<BagOfPattern>();
                split.put(label, class_set);
            }

            split.get(label).add(transformed_data.get(i));
        }

        return split;
    }

    @Override
    public void train_binary(TSDataset data) throws Exception {

    }

    @Override
    public TIntObjectMap<TSDataset> split(TSDataset data, int[] indices) throws Exception {
        long startTime = System.nanoTime();
        this.node.tree.stats.boss_splitter_train_time += (System.nanoTime() - startTime);
        return best_splits;
    }

    @Override
    public int predict(TimeSeries query, int queryIndex) throws Exception {

        return 0;

//        SAXTransformerContainer sax_transform = ((SAXTransformerContainer)this.node.tree.getForest().getTransforms().get("sax"));
//        HashMap<String, Double> query_hist = sax_transform.ts_to_bag(saxParams, query);
//
////        closest_nodes.clear();
//
//        List<Integer> closest_nodes = new ArrayList<Integer>();
//        double minDistance = Double.MAX_VALUE;
//        int min_key = -1;
//        double distance = 0;
//        for (int key : bopPerClass.keys()) {
//            BagOfPattern example_hist = bopPerClass.get(key);
//
//            distance = SaxDistance_map(query_hist, example_hist.getBag());
//
//            if (distance < minDistance) {
//                minDistance = distance;
//                min_key = key;
//                closest_nodes.clear();
//                closest_nodes.add(min_key);
//            }else if (distance == minDistance) {
////				if (distance == min_distance) {
////					System.out.println("min distances are same " + distance + ":" + min_distance);
////				}
//                minDistance = distance;
//                closest_nodes.add(key);
//            }
//        }
//
//
//        int r = ThreadLocalRandom.current().nextInt(closest_nodes.size());	//TODO may be use SplitRandom??
//        return closest_nodes.get(r);

    }

}
