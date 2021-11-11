package trees.splitters;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;

import com.carrotsearch.hppc.cursors.IntIntCursor;

import com.sun.imageio.plugins.common.BogusColorSpace;
import core.AppContext;
import core.ParallelFor;
import core.AppContext.SplitMethod;
import core.AppContext.SplitterType;
import core.AppContext.TransformLevel;
import datasets.BossDataset;
import datasets.BossDataset.BossParams;
import datasets.TSDataset;
import datasets.TimeSeries;
import dev.BossTransformContainer;
import dev.Classifier.Predictions;
import feature.FeatureSFA;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import org.omg.PortableInterceptor.SYSTEM_EXCEPTION;
import transforms.BOSS;
import transforms.SFA;
import transforms.BOSS.BagOfPattern;
import transforms.SFA.HistogramType;
import transforms.SFA.Words;
import trees.ProximityTree.Node;
import util.FeatureUtil;
import util.Sampler;
import util.Util;
import util.pair.Pair;

public class BossSplitter implements NodeSplitter {

	TIntObjectHashMap<BagOfPattern> bopPerClass;

	public TIntObjectHashMap<Pair<FeatureSFA, ArrayList<Integer>>> bestSFA_PerClass;

	int windowLength = 8;
	int wordLength = 4;
	int symbols = 4;
	boolean normMean = true;
	public static boolean[] NORMALIZATION = new boolean[] { true, false };
	
	public Node node;
	TIntObjectMap<TSDataset> best_splits = null; //TODO

    double train_node_time = 0.0;

	public BossParams getForest_transform_params() {
		return forest_transform_params;
	}

	BossParams forest_transform_params;
	
	public BossSplitter(Node node) {
		// TODO Auto-generated constructor stub
		this.node = node;
		this.node.tree.stats.boss_count++;

//		sfa_transforms = ((BossTransformContainer)this.node.tree.getForest().getTransforms().get("boss")).getSFATransforms();
	}

	public int getWindowLength() {
		return windowLength;
	}

	public int getWordLength() {
		return wordLength;
	}

	public int getSymbols() {
		return symbols;
	}

	public boolean isNormMean() {
		return normMean;
	}

	@Override
	public TIntObjectMap<TSDataset> train(TSDataset data, int[] indices) throws Exception {
		long startTime = System.nanoTime();

		boolean sfa_ok = train_using_forest_transform(data, indices);
		if(!sfa_ok) return null;

		this.node.tree.stats.boss_splitter_train_time += (System.nanoTime() - startTime);
		
		return split(data, indices);
	}


	@Override
	public void train_binary(TSDataset data) throws Exception {
		// TODO Auto-generated method stub
		
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

	public boolean check_sfa_sp_valid_with_dis(ArrayList<Integer> ls, int threshold, ArrayList<Integer> test_label, int seriesLabel){
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

	public Pair<FeatureSFA, ArrayList<Integer>> getBestSfa_from_example(BagOfPattern example_hist,ArrayList<BagOfPattern> node_boss_dataset, ArrayList<Integer> node_boss_dataset_label){

		int seriesId = example_hist.getSeriesId();
		int seriesLabel = example_hist.getLabel();
		ArrayList<String> example_sfa_ls = example_hist.sfa_ls;

		//选出信息增益最大的sfa
		String best_sfaword_string = null;
		int best_sfaword_val = 0;
		int best_sfaword_id = 0;
		double infogain_max = -1;
		int best_sp_threshold = 0;
		ArrayList<Integer> best_sfa_dis_ls = null;

        HashMap<String, FeatureSFA> sfaword_to_feature = new HashMap<>();
        ArrayList<FeatureSFA> best_sfa_ls = new ArrayList<>();

		int example_sfa_size = example_sfa_ls.size();
		String pre_sfa_word = null;
		for (int sfa_word_idx = 0; sfa_word_idx < example_sfa_size; sfa_word_idx++) {

			String example_sfa_string = example_sfa_ls.get(sfa_word_idx);
			if(example_sfa_string.equals(pre_sfa_word)){
				continue;
			}
			if(sfaword_to_feature.containsKey(example_sfa_string)){
                FeatureSFA best_sfa_tmp = sfaword_to_feature.get(example_sfa_string);
                if(best_sfa_tmp==null){
                    continue;
                }
                best_sfa_ls.add(new FeatureSFA(-1, "SFA", seriesLabel, best_sfa_tmp.infogain, best_sfa_tmp.threshold, 0, best_sfa_tmp.SFAvalue, best_sfa_tmp.SFA_abc, forest_transform_params.window_len, seriesId, forest_transform_params, sfa_word_idx));
                continue;
            }

			pre_sfa_word = example_sfa_string;
			int example_sfaword_len = example_sfa_string.length();

			//该sax_word与每个数据之间的距离
			ArrayList<Integer> sfa_dis_ls = new ArrayList<>();
			for (BagOfPattern boss_dataset_tmp: node_boss_dataset) {
				ArrayList<String> ts_sfaword_ls = boss_dataset_tmp.sfa_ls;
				HashSet<String> ts_sfaword_set = new HashSet<>(ts_sfaword_ls);
				int dis_min = Integer.MAX_VALUE;

				for (String sfaword_tmp : ts_sfaword_set) {

					int dis_tmp = 0;
					for (int i = 0; i < example_sfaword_len; i++) {
						dis_tmp += Math.abs(example_sfa_string.charAt(i)-sfaword_tmp.charAt(i));
					}
					if(dis_min>dis_tmp){
						dis_min = dis_tmp;
					}
				}
				sfa_dis_ls.add(dis_min);
			}
			Pair<Double, Integer> infoGain_tmp = FeatureUtil.getInfoGain(sfa_dis_ls, node_boss_dataset_label, seriesLabel);

			if(infoGain_tmp.getKey()>=infogain_max){

				best_sp_threshold = infoGain_tmp.getValue();
				best_sfaword_id = sfa_word_idx;
				best_sfaword_string = example_sfa_string;
				best_sfa_dis_ls = sfa_dis_ls;

				if(!check_sfa_sp_valid_with_dis(best_sfa_dis_ls, best_sp_threshold, node_boss_dataset_label, seriesLabel)){
					continue;
				}

				if(infoGain_tmp.getKey()>infogain_max){

					best_sfa_ls.clear();
				}

				infogain_max = infoGain_tmp.getKey();
                FeatureSFA best_sfa_tmp = new FeatureSFA(-1, "SFA", seriesLabel, infogain_max, best_sp_threshold, 0, best_sfaword_val, best_sfaword_string, forest_transform_params.window_len, seriesId, forest_transform_params, best_sfaword_id);
                best_sfa_ls.add(best_sfa_tmp);
                sfaword_to_feature.put(best_sfaword_string, best_sfa_tmp);
            }else {
                sfaword_to_feature.put(best_sfaword_string, null);
            }
		}

		int best_sfa_ls_size = best_sfa_ls.size();
		if(best_sfa_ls_size == 0){
			return null;
		}
		Random r = new Random();
		int choose_best_id = r.nextInt(best_sfa_ls_size);

		//后面在遍历时填入featureID
		FeatureSFA best_sfa = best_sfa_ls.get(choose_best_id);
        return new Pair<>(best_sfa, best_sfa_dis_ls);

	}
	
	private boolean train_using_forest_transform(TSDataset data, int[] indices) throws Exception {

	    train_node_time = 0;

		long startTime = System.nanoTime();
		
		BossTransformContainer transforms = ((BossTransformContainer)this.node.tree.getForest().getTransforms().get("boss"));
		
		//pick a random tansform
		SplittableRandom rand = new SplittableRandom();
		int r = rand.nextInt(transforms.boss_params.size());
		forest_transform_params = transforms.boss_params.get(r);
		
		BossDataset transformed_dataset = transforms.boss_datasets.get(forest_transform_params.toString());
		BagOfPattern[] transformed_data = transformed_dataset.getTransformed_data();

		//form the node dataset
		ArrayList<BagOfPattern> node_boss_dataset = new ArrayList<BagOfPattern>();
		for (int j = 0; j < indices.length; j++) {
			node_boss_dataset.add(transformed_data[indices[j]]);
		}

		ArrayList<Integer> node_boss_dataset_label = new ArrayList<>();
		for (BagOfPattern bag : node_boss_dataset) {
			node_boss_dataset_label.add(bag.getLabel());
		}

		//split by class
		TIntObjectMap<List<BagOfPattern>> boss_data_per_class = split_by_class(node_boss_dataset);

		long preTime = System.nanoTime()-startTime;

		bopPerClass = new TIntObjectHashMap<>();
		bestSFA_PerClass = new TIntObjectHashMap<>();

		boolean flag = true;

		for (int key : boss_data_per_class.keys()) {
			BagOfPattern example;
			Pair<FeatureSFA, ArrayList<Integer>> bestSfa_from_example;
			r = rand.nextInt(boss_data_per_class.get(key).size());
			example = boss_data_per_class.get(key).get(r);

			bestSfa_from_example = getBestSfa_from_example(example, node_boss_dataset, node_boss_dataset_label);
			if(bestSfa_from_example==null){
				return false;
			}

			bopPerClass.put(key, example);
			bestSFA_PerClass.put(key, bestSfa_from_example);

			if(flag){
				int example_sfa_num = example.sfa_ls.size();
				int data_num = node_boss_dataset.size();

				flag = false;

                int word_length = example.sfa_ls.get(0).length();
                int wordset_m = node_boss_dataset.get(0).sfa_ls.size();
                double time = (word_length/8.0)*(wordset_m/116.0)* 0.000089 * example_sfa_num*data_num*(boss_data_per_class.keys().length);
                train_node_time = time;
			}

		}

		//根据最佳SFA和train数据的距离分类
		//for each instance at node, find the closest distance to examplar
		int d = Integer.MAX_VALUE;
		TIntObjectMap<TSDataset> splits = new TIntObjectHashMap<TSDataset>();

		startTime = System.nanoTime();

		for (int j = 0; j < node_boss_dataset.size(); j++) {

			double maxInfogain = -1.0;
			int min_key = -1;
			closest_nodes.clear();
			double d_min = Double.MAX_VALUE;
			int d_min_key = -1;

			for (int key : bestSFA_PerClass.keys()) {
				splits.putIfAbsent(key, new TSDataset());

				d = bestSFA_PerClass.get(key).getValue().get(j);
				FeatureSFA featureSFA = bestSFA_PerClass.get(key).getKey();

				if (d < featureSFA.threshold) {
					if(featureSFA.infogain>maxInfogain){
						maxInfogain = featureSFA.infogain;
						closest_nodes.clear();
						closest_nodes.add(key);
					}else if(featureSFA.infogain==maxInfogain){
						closest_nodes.add(key);
					}
				}

				if(d-featureSFA.threshold<d_min){
					d_min = d-featureSFA.threshold;
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
			splits.get(min_key).add(node_boss_dataset.get(j).getSeries());

		}

		train_node_time += ((System.nanoTime() - startTime)/AppContext.NANOSECtoMILISEC);
		
		//TODO temp
		this.best_splits = splits;

		return true;

	} //end func


	public static BagOfPattern createBagOfPattern(final int[] words, final TimeSeries sample, final int wordLength, int symbols, int queryIndex,ArrayList<String> sfaWords) {
		BagOfPattern bagOfPatterns;

		final byte usedBits = (byte) Words.binlog(symbols);
		// FIXME
		// final long mask = (usedBits << wordLength) - 1l;
		final long mask = (1L << (usedBits * wordLength)) - 1L;

		// iterate all samples
		bagOfPatterns = new BagOfPattern(words.length, sample, queryIndex, sfaWords);
																						

		// create subsequences
//		long lastWord = Long.MIN_VALUE;
//
//		for (int offset = 0; offset < words.length; offset++) {
//			// use the words of larger queryLength to get words of smaller lengths
//			long word = words[offset] & mask;
//			if (word != lastWord) { // ignore adjacent samples
//
//				bagOfPatterns.bag.putOrAdd((int) word, (short) 1, (short) 1);
//			}
//			lastWord = word;
//		}
//		System.out.println(bagOfPatterns.bag);

		return bagOfPatterns;
	}
	
	  /**
	   * Create the BOSS boss for a fixed window-queryLength and SFA word queryLength
	   *
	   * @param words      the SFA words of the time series
	   * @param samples    the samples to be transformed
	   * @param wordLength the SFA word queryLength
	   * @return returns a BOSS boss for each time series in samples
	   */
	  public static BagOfPattern[] createBagOfPattern(
	      final int[][] words,
	      final TimeSeries[] samples,
	      final int wordLength,
	      final int symbols,
		  ArrayList<ArrayList<String>> dataset_sfawordLs) {
	    BagOfPattern[] bagOfPatterns = new BagOfPattern[words.length];

//	    final byte usedBits = (byte) Words.binlog(symbols);
	    // FIXME
	    // final long mask = (usedBits << wordLength) - 1l;
//	    final long mask = (1L << (usedBits * wordLength)) - 1L;

	    // iterate all samples
	    for (int j = 0; j < words.length; j++) {
//	      bagOfPatterns[j] = new BagOfPattern(words[j].length, samples[j].getLabel());
	      bagOfPatterns[j] = new BagOfPattern(words[j].length, samples[j],j, dataset_sfawordLs.get(j)); //TODO changed to time series

	      // create subsequences
//	      long lastWord = Long.MIN_VALUE;
//
//	      for (int offset = 0; offset < words[j].length; offset++) {
//	        // use the words of larger queryLength to get words of smaller lengths
//	        long word = words[j][offset] & mask;
//	        if (word != lastWord) { // ignore adjacent samples
//
//	          bagOfPatterns[j].getBag().putOrAdd((int) word, (short) 1, (short) 1);
//	        }
//	        lastWord = word;
//	      }
	    }

	    return bagOfPatterns;
	  }
	
	
	//TODO NOTE this method is different to train because data subset may be used to train
	@Override
	public TIntObjectMap<TSDataset> split(TSDataset data, int[] indices) throws Exception {
		long startTime = System.nanoTime();

		//time is measured separately for split function from train function as it can be called separately -- to prevent double counting 
		this.node.tree.stats.boss_splitter_train_time += (System.nanoTime() - startTime);
		
		return best_splits;	//TODO
	}

	List<Integer> closest_nodes = new ArrayList<Integer>();

	@Override
	public int predict(TimeSeries query, int queryIndex) throws Exception {

		return 0;
//		long minDistance = Integer.MAX_VALUE;
//		long distance = Integer.MAX_VALUE;
//		int min_key = 0; //TODO
//
//		if (AppContext.boss_split_method == SplitMethod.Binary_Gini) {
//			return predict_gini_split(query, queryIndex);
//		}
//
//		BagOfPattern query_hist;
//
//		if (AppContext.boss_transform_level == TransformLevel.Forest) {
//			BossTransformContainer transforms = ((BossTransformContainer)this.node.tree.getForest().getTransforms().get("boss"));
//			query_hist = transforms.transform_series_using_sfa(query, transforms.sfa_transforms.get(forest_transform_params.toString()), queryIndex);
//		}else {
//			query_hist = this.node.tree.boss_test_dataset_tree_level.getTransformed_data()[queryIndex];
//		}
//
//		closest_nodes.clear();
//
//
//		for (int key : bopPerClass.keys()) {
//			BagOfPattern example_hist = bopPerClass.get(key);
//
//			distance = BossDistance(query_hist, example_hist);
//
//			if (distance < minDistance) {
//				minDistance = distance;
//				min_key = key;
//				closest_nodes.clear();
//				closest_nodes.add(min_key);
//			}else if (distance == minDistance) {
////				if (distance == min_distance) {
////					System.out.println("min distances are same " + distance + ":" + min_distance);
////				}
//				minDistance = distance;
//				closest_nodes.add(key);
//			}
//		}
//
//
//		int r = ThreadLocalRandom.current().nextInt(closest_nodes.size());	//TODO may be use SplitRandom??
//		return closest_nodes.get(r);
	}


	public TIntObjectHashMap<BagOfPattern> getBopPerClass() {
		return bopPerClass;
	}


	protected boolean compareLabels(Double label1, Double label2) {
		// compare 1.0000 to 1.0 in String returns false, hence the conversion to double
		return label1 != null && label2 != null && label1.equals(label2);
	}
	
	public String toString() {
		return "BossSplitter[ForestParam:" + forest_transform_params+ "]";
	}
	
}
