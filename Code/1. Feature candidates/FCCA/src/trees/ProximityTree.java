package trees;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

import core.AppContext;
import core.TreeStatCollector;
import core.AppContext.ParamSelection;
import datasets.*;
import datasets.BossDataset.BossParams;
import dev.BossTransformContainer;
import dev.RIFTransformContainer;
import dev.SAXTransformerContainer;
import feature.*;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import transforms.RIF;
import transforms.SFA;
import transforms.BOSS.BagOfPattern;
import transforms.SFA.HistogramType;
import transforms.SFA.Words;
import trees.splitters.*;
import trees.ProximityForest.Predictions;
import util.Util;
import util.pair.Pair;


/**
 * 
 * @author shifaz
 * @email ahmed.shifaz@monash.edu
 *
 */

public class ProximityTree{
	protected int forest_id;	//TODO remove
	private int tree_id;
	protected Node root;
	protected int node_counter = 0;
	
	protected transient Random rand;
	public TreeStatCollector stats;
	protected transient ProximityForest forest;	//NOTE

	public double beta = 1e-10; // for boosting computation
	
	//TODO after training deallocate this
	public transient BossDataset boss_train_dataset_tree_level; //FIXME done at root node for all datasets, indices should match with original dataset
	public transient BossDataset boss_test_dataset_tree_level; //FIXME done at root node for all datasets, indices should match with original dataset

	public transient BossDataset.BossParams training_params;	//use this to transform test dataset
	public transient SFA training_sfa;

	public transient RIFTransformContainer rif_transfomer;
	public transient TSDataset rif_train_data;
	public transient TSDataset rif_test_data;
	
	//TODO temp keeping per tree because if we choose to do boosting this set may differ from the dataset at forest level
	// better to use indices?
	public transient TSDataset train_data;
	
	protected transient DataStore treeDataStore;

	public transient double[] SFA_alpha_weight = {1.0, 0.5, 0.25, 0.12, 0.06, 0.03, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

	public ProximityTree(int tree_id, ProximityForest forest) {
		this.forest = forest;
		this.forest_id = forest.forest_id;
		this.tree_id = tree_id;
		this.rand = ThreadLocalRandom.current();
		stats = new TreeStatCollector(forest_id, tree_id);
		treeDataStore = new DataStore();
	}
	
	public ProximityForest getForest() {
		return this.forest;
	}

	public Node getRootNode() {
		return this.root;
	}
	
	private BossDataset boss_transform_at_tree_level(TSDataset data, boolean train_sfa) throws Exception {
		//if transformations are enabled perform transformation here
		//select parameters for the transformation from random sampling of CV params obtained before
		//param selection is per tree
		

		//transform the dataset using the randomly picked set of params;
		
		TimeSeries[] samples = data._get_internal_list().toArray(new TimeSeries[] {}); //FIXME
		
		if (train_sfa == true) {
			bossParamSelect(data, AppContext.boss_param_selection);
			System.out.println("boss training params for tree: " + this.tree_id + " " + training_params);
			
			training_sfa = new SFA(HistogramType.EQUI_DEPTH); 
			training_sfa.fitWindowing(samples, training_params.window_len, training_params.word_len, training_params.alphabet_size, training_params.normMean, training_params.lower_bounding);
		}
		
		final int[][] words = new int[samples.length][];
		ArrayList<ArrayList<String>> dataset_sfawordLs = new ArrayList<>();
		
        for (int i = 0; i < samples.length; i++) {
              short[][] sfaWords = training_sfa.transformWindowing(samples[i]);
			ArrayList<String> sfaword_ls_tmp = new ArrayList<>();
              words[i] = new int[sfaWords.length];
              for (int j = 0; j < sfaWords.length; j++) {
				  sfaword_ls_tmp.add(Arrays.toString(sfaWords[j]));
                words[i][j] = (int) Words.createWord(sfaWords[j], training_params.word_len, (byte) Words.binlog(training_params.alphabet_size));
              }
              dataset_sfawordLs.add(sfaword_ls_tmp);
          }

		BagOfPattern[] histograms = BossSplitter.createBagOfPattern(words, samples, training_params.word_len, training_params.alphabet_size, dataset_sfawordLs);
		
		return new BossDataset(data, histograms, training_params);
	}
	
	public void train(TSDataset data, int treeID) throws Exception {

		if (AppContext.boss_enabled & (AppContext.boss_transform_level == AppContext.TransformLevel.Forest)) {
			//it must be done at forest level
		}else if (AppContext.boss_enabled & (AppContext.boss_transform_level == AppContext.TransformLevel.Tree)) {
			boss_train_dataset_tree_level = boss_transform_at_tree_level(data, true);
		}else if (AppContext.boss_enabled) {
			throw new Exception("ERROR: transformation level not supported for boss");
		}
		
		if (AppContext.rif_enabled) {
			rif_transfomer = new RIFTransformContainer(1);
			rif_transfomer.fit(data);
			rif_train_data = rif_transfomer.transform(data);
		}
		
		this.train_data = data; //keeping a reference to this //TODO used to extract indices at node level, quick fix
		
		this.root = new Node(null, null, ++node_counter, this);
		this.root.class_distribution_str = data.get_class_map().toString();		//TODO debug only - memory leak

		this.root.train(data);

	}
	
	
	private BossDataset.BossParams bossParamSelect(TSDataset data, ParamSelection method) throws Exception {
		BossDataset.BossParams boss_params = null;
		int[] word_len = BossTransformContainer.getBossWordLengths();		

		SplittableRandom rand = new SplittableRandom();
		
		if (method == ParamSelection.Random) {
			
			
			//choose random params
//			Random rand = ThreadLocalRandom.current();
			
			boolean norm = rand.nextBoolean();			
			int w = rand.nextInt(10, data.length());
			int l = word_len[rand.nextInt(word_len.length)];
			
			boss_params = new BossParams(norm, w, l, 4);
			
		}else if (method == ParamSelection.PreLoadedSet){
			//choose best from predefined values

			List<BossDataset.BossParams> best_boss_params = AppContext.boss_preloaded_params.get(AppContext.getDatasetName());
			int r2  = rand.nextInt(best_boss_params.size()); //pick a random set of params among the best params for this datatset
			boss_params = best_boss_params.get(r2);
			
		}else {
			throw new Exception("Boss param selection method not supported");
		}
		
		
		this.training_params = boss_params;	//store to use during testing
		return boss_params;
	}
	
	public Predictions predict(TSDataset test) throws Exception {
		Predictions predctions = forest.new Predictions(test.size()); //TODO best memory management?
		
		//TODO do transformation of dataset here
		//if transformations are enabled perform transformation here
		//select parameters for the transformation from random sampling of CV params obtained before
		//param selection is per tree
		if (AppContext.boss_enabled & (AppContext.boss_transform_level == AppContext.TransformLevel.Forest)) {
			//it must be done at forest level
		}else if (AppContext.boss_enabled & (AppContext.boss_transform_level == AppContext.TransformLevel.Tree)) {
			boss_test_dataset_tree_level = boss_transform_at_tree_level(test, false);
		}else if (AppContext.boss_enabled){
			throw new Exception("ERROR: transformation level not supported for boss");
		}
		
		if (AppContext.rif_enabled) {
			rif_test_data = rif_transfomer.transform(test);
		}
		
		for (int i = 0; i < test.size(); i++) {
			TimeSeries query = test.get_series(i);
			//System.out.println("***test:"+i+"***");
			Integer predicted_label = predict(query, i);
			//System.out.println("pred:"+predicted_label);
			
			//TODO assert predicted_label = null;
			if (predicted_label == null) {
				throw new Exception("ERROR: possible bug detected: predicted label is null");
			}
			
			predctions.predicted_labels[i] = predicted_label;
			
			//TODO object equals or == ??
			if (predicted_label.equals(query.getLabel())) {
				predctions.correct.incrementAndGet();	//NOTE accuracy for tree
			}
		}
		
		return predctions;
	}


	private double getEntropy(ArrayList<Integer> label_ls, int label){
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

	private Pair<Double, Integer> getInfoGain(ArrayList<Integer> sfaList, ArrayList<Integer> test_label, int label){
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

	private Pair<Double, Double> getInfoGain_double(ArrayList<Double> sfaList, ArrayList<Integer> test_label, int label){
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

	public class rise{
		public int featureId;
		public double InfoGain;
		public int attribute;
		public double threshold;
		public int threshold_sign;
		public int rise2cls;

		public rise(int featureId, double infoGain, int attribute, double threshold, int threshold_sign, int rise2cls) {
			this.featureId = featureId;
			InfoGain = infoGain;
			this.attribute = attribute;
			this.threshold = threshold;
			this.threshold_sign = threshold_sign;
			this.rise2cls = rise2cls;
		}
	}

	public FeatureRISE get_best_FeatureRISE(ArrayList<Double> train_rise_ls, double threshold, ArrayList<Integer> test_label, int[] unique_label, int featureID, RIFSplitter rifSplitter, int bestAttribute){

		double best_infogain = 0;
		int best_label=100000;
		int best_threshold_sign=1;
		double best_threshold=threshold;

		for (int label : unique_label) {
			//计算当前label下的信息增益
			double entropy_origin = getEntropy(test_label, label);

			ArrayList<Integer> l_label_ls=new ArrayList<>();
			ArrayList<Integer> r_label_ls=new ArrayList<>();
			for (int i = 0; i < train_rise_ls.size(); i++) {
				double x = train_rise_ls.get(i);
				if(x>=threshold) r_label_ls.add(test_label.get(i));
				else l_label_ls.add(test_label.get(i));
			}

			double entropy1 = getEntropy(l_label_ls, label);
			double entropy2 = getEntropy(r_label_ls, label);

			double sp_infogain = entropy_origin - (l_label_ls.size()*1.0/test_label.size())*entropy1 - (r_label_ls.size()*1.0/test_label.size())*entropy2;
			if(sp_infogain>=best_infogain){
				best_infogain=sp_infogain;

				//找到对应label的符号（大于该阈值，取该label OR 小于该阈值，取该label）
				int threshold_sign = 0;
				int large_th_cnt = 0, less_th_cnt = 0;
				for (int i = 0; i < train_rise_ls.size(); i++) {
					if(train_rise_ls.get(i)>=threshold && test_label.get(i)==label){
						large_th_cnt++;
					}
					if(train_rise_ls.get(i)<threshold && test_label.get(i)==label){
						less_th_cnt++;
					}
				}
				if(large_th_cnt>=less_th_cnt) threshold_sign=1;
				else threshold_sign=0;

				best_threshold_sign = threshold_sign;
				best_label = label;
			}
		}

		RIF rif_transformer = rifSplitter.getRif_transformer();

		return new FeatureRISE(featureID, "RISE", best_label, best_infogain, best_threshold, best_threshold_sign,
								bestAttribute, rif_transformer.getIntervals().get(0)[0], rif_transformer.getIntervals().get(0)[1], rif_transformer.getFilter_type().toString());
	}

	public boolean check_sax_sp_valid(ArrayList<Double> ls, double threshold, ArrayList<Integer> test_label, int seriesLabel){
		int large_th_cnt = 0, less_th_cnt = 0;
		for (int i = 0; i < ls.size(); i++) {
			if(ls.get(i)>=threshold && test_label.get(i)==seriesLabel){
				large_th_cnt++;
			}
			if(ls.get(i)<threshold && test_label.get(i)==seriesLabel){
				less_th_cnt++;
			}
		}
		if(large_th_cnt<less_th_cnt) return false;
		return true;
	}

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

    public boolean check_sfa_sp_valid(ArrayList<Integer> ls, int threshold, ArrayList<Integer> test_label, int seriesLabel){
        int large_th_cnt = 0, less_th_cnt = 0;
        for (int i = 0; i < ls.size(); i++) {
            if(ls.get(i)>=threshold && test_label.get(i)==seriesLabel){
                large_th_cnt++;
            }
            if(ls.get(i)<threshold && test_label.get(i)==seriesLabel){
                less_th_cnt++;
            }
        }
        if(large_th_cnt<less_th_cnt) return false;
        return true;
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


	//一共off位
    public String intToBinary(int data, int off){

		StringBuffer buffer=new StringBuffer();
		String binary_h="0";
		String binary_o= Integer.toBinaryString(data);
		if(binary_o.toCharArray().length<off) {
			binary_h=new String(new char[off-binary_o.length()]).replace("\0", binary_h);//重复binsry_h
			buffer.append(binary_h+binary_o);
			return buffer.toString();
		}else {

			buffer.append(binary_o);
			return buffer.toString();
		}

	}

    public short[] changeSfavalToSfachar(int sfaWord_value, int sfaWord_length){
		int negative_flag = 0;
		if(sfaWord_value<0){
			sfaWord_value=-sfaWord_value;
			negative_flag=1;
		}

		String sfaWord_binary = intToBinary(sfaWord_value, sfaWord_length*2);

		if(negative_flag==1){
			StringBuffer sfaWord_binary_not = new StringBuffer();
			for (int i = 0; i < sfaWord_binary_not.length(); i++) {
				if(sfaWord_binary_not.charAt(i)=='1'){
					sfaWord_binary_not.append('0');
				}else {
					sfaWord_binary_not.append('1');
				}
			}

			sfaWord_binary_not.insert(0,'0');
			int sfaWord_binary_not_int = Integer.parseInt(sfaWord_binary_not.toString(), 2);
			sfaWord_binary_not_int+=1;

			sfaWord_binary = intToBinary(sfaWord_binary_not_int, sfaWord_length*2);
			sfaWord_binary = '1'+sfaWord_binary.substring(1);
		}

		ArrayList<Integer> sfaWord_list = new ArrayList<>();
		for (int i = 0; i < sfaWord_length; i++) {
			int sfaWord_tmp = Integer.parseInt(sfaWord_binary.substring(i * 2, i * 2 + 2), 2);
			sfaWord_list.add(sfaWord_tmp);
		}
		Collections.reverse(sfaWord_list);

		short[] result = new short[sfaWord_list.size()];
		for (int i = 0; i < sfaWord_list.size(); i++) {
			result[i] = (short) (sfaWord_list.get(i).intValue());
		}

		return result;
	}


	public ProximityForest.SingletreeFeatureLs get_singletree_feature_distance_parallel(TSDataset train_data, TSDataset test_data, TSDataset validate_data, int all_feature_num){

		ArrayList<FeatureSFA> sfa_feature = new ArrayList<>();
		ArrayList<ArrayList<Integer>> sfa_train = new ArrayList<>();
        ArrayList<ArrayList<Integer>> sfa_validate = new ArrayList<>();
		ArrayList<ArrayList<Integer>> sfa_test = new ArrayList<>();

		ArrayList<FeatureRISE> rise_feature = new ArrayList<>();
		ArrayList<ArrayList<Double>> rise_train = new ArrayList<>();
        ArrayList<ArrayList<Double>> rise_validate = new ArrayList<>();
		ArrayList<ArrayList<Double>> rise_test = new ArrayList<>();

		ArrayList<FeatureSAX> sax_feature = new ArrayList<>();
		ArrayList<ArrayList<Double>> sax_train = new ArrayList<>();
        ArrayList<ArrayList<Double>> sax_validate = new ArrayList<>();
		ArrayList<ArrayList<Double>> sax_test = new ArrayList<>();

		ArrayList<Feature> allFeatureList = new ArrayList<>();

		ArrayList<Integer> train_label = new ArrayList<>();
		for (int i = 0; i < train_data.size(); i++) {
			train_label.add(train_data.get_series(i).getLabel());
		}

		Node node = this.root;
		Stack<Node> st=new Stack<>();
		st.push(node);

		int[] unique_classes = train_data.get_unique_classes();

		while(!st.empty()){
			Node tmpnode = st.peek();
			st.pop();
			if(tmpnode==null || tmpnode.is_leaf()){
				continue;
			}

			NodeSplitter best_splitter = tmpnode.splitter.getBest_splitter();
			if(best_splitter instanceof RIFSplitter){

				long rise_st_time = System.nanoTime();

				double bestThreshold = ((RIFSplitter) best_splitter).getBinary_splitter().getBestThreshold();
				int bestAttribute = ((RIFSplitter) best_splitter).getBinary_splitter().getBestAttribute();

				ArrayList<Double> train_rise_ls = new ArrayList<>();
                ArrayList<Double> validate_rise_ls = new ArrayList<>();
				ArrayList<Double> test_rise_ls = new ArrayList<>();

				for (int i = 0; i < train_data.size(); i++) {
                    TimeSeries ts = train_data.get_series(i);
                    double v = ((RIFSplitter) best_splitter).getattr_withVal(ts, i);
                    train_rise_ls.add(v);
                }

                for (int i = 0; i < validate_data.size(); i++) {
                    TimeSeries ts = validate_data.get_series(i);
                    double v = ((RIFSplitter) best_splitter).getattr_withVal(ts, i);
                    validate_rise_ls.add(v);
                }

				for (int i = 0; i < test_data.size(); i++) {
					TimeSeries ts = test_data.get_series(i);
					double v = ((RIFSplitter) best_splitter).getattr_withVal(ts, i);
					test_rise_ls.add(v);
				}

				FeatureRISE best_rise = get_best_FeatureRISE(train_rise_ls, bestThreshold, train_label, train_data.get_unique_classes(), all_feature_num, (RIFSplitter) best_splitter,bestAttribute);
				rise_feature.add(best_rise);
				allFeatureList.add(new Feature(best_rise.featureId, best_rise.featureType, best_rise.representLabel, best_rise.infogain, best_rise.threshold, best_rise.sign));

				ArrayList<Double> tmp = new ArrayList<>();
				tmp.add((double)all_feature_num);
				tmp.addAll(train_rise_ls);
				rise_train.add(tmp);

                ArrayList<Double> tmp2 = new ArrayList<>();
                tmp2.add((double)all_feature_num);
                tmp2.addAll(validate_rise_ls);
                rise_validate.add(tmp2);

				ArrayList<Double> tmp1 = new ArrayList<>();
				tmp1.add((double)all_feature_num);
				tmp1.addAll(test_rise_ls);
				rise_test.add(tmp1);

				all_feature_num++;

			}
			else if(best_splitter instanceof SaxSplitter){
				SaxDataset.SAXParams saxParams = ((SaxSplitter) best_splitter).saxParams;
				TIntObjectHashMap<Pair<FeatureSAX, ArrayList<Double>>> bestSAX_perClass = ((SaxSplitter) best_splitter).bestSAX_PerClass;
				SAXTransformerContainer sax_transform = (SAXTransformerContainer)(((SaxSplitter) best_splitter).node.tree.getForest().getTransforms().get("sax"));

				for (int class_key : bestSAX_perClass.keys()) {
					FeatureSAX best_sax = bestSAX_perClass.get(class_key).getKey();
					best_sax.featureId = all_feature_num;
					String best_saxword_string = best_sax.sax_word;

					sax_feature.add(best_sax);

					allFeatureList.add(new Feature(best_sax.featureId, best_sax.featureType, best_sax.representLabel, best_sax.infogain, best_sax.threshold, best_sax.sign));

					ArrayList<Double> best_sax_dis_ls_test = new ArrayList<>();
					int best_saxword_len = best_saxword_string.length();
					for (int j = 0; j < test_data.size(); j++) {
						TimeSeries query = test_data.get_series(j);
						ArrayList<String> query_saxword_ls = sax_transform.ts_to_saxwordls(saxParams, query);

						HashSet<String> ts_saxword_set = new HashSet<>(query_saxword_ls);
						double dis_min = Double.MAX_VALUE;

						for (String saxword_tmp : ts_saxword_set) {

							double dis_tmp = 0;
							for (int i = 0; i < best_saxword_len; i++) {
								dis_tmp += Math.abs(best_saxword_string.charAt(i)-saxword_tmp.charAt(i));
							}
							if(dis_min>dis_tmp){
								dis_min = dis_tmp;
							}
						}

						best_sax_dis_ls_test.add(dis_min);
					}
					ArrayList<Double> tmp3 = new ArrayList<>();
					tmp3.add((double)all_feature_num);
					tmp3.addAll(best_sax_dis_ls_test);
					sax_test.add(tmp3);

                    ArrayList<Double> best_sax_dis_ls_validate = new ArrayList<>();
                    for (int j = 0; j < validate_data.size(); j++) {
                        TimeSeries query = validate_data.get_series(j);
                        ArrayList<String> query_saxword_ls = sax_transform.ts_to_saxwordls(saxParams, query);

                        HashSet<String> ts_saxword_set = new HashSet<>(query_saxword_ls);
                        double dis_min = Double.MAX_VALUE;

                        for (String saxword_tmp : ts_saxword_set) {

                            double dis_tmp = 0;
                            for (int i = 0; i < best_saxword_len; i++) {
                                dis_tmp += Math.abs(best_saxword_string.charAt(i)-saxword_tmp.charAt(i));
                            }
                            if(dis_min>dis_tmp){
                                dis_min = dis_tmp;
                            }
                        }

                        best_sax_dis_ls_validate.add(dis_min);
                    }
                    ArrayList<Double> tmp4 = new ArrayList<>();
                    tmp4.add((double)all_feature_num);
                    tmp4.addAll(best_sax_dis_ls_validate);
                    sax_validate.add(tmp4);

					all_feature_num++;

				}
			}
			else if(best_splitter instanceof BossSplitter){

				TIntObjectHashMap<BagOfPattern> bopPerClass = ((BossSplitter) best_splitter).getBopPerClass();
				BossParams bossParams = ((BossSplitter) best_splitter).getForest_transform_params();
				TIntObjectHashMap<Pair<FeatureSFA, ArrayList<Integer>>> bestSFA_perClass = ((BossSplitter) best_splitter).bestSFA_PerClass;
				BossTransformContainer boss_transform = (BossTransformContainer)(((BossSplitter) best_splitter).node.tree.getForest().getTransforms().get("boss"));
				SFA sfa = boss_transform.sfa_transforms.get(bossParams.toString());

				for(int class_key: bestSFA_perClass.keys()){

					FeatureSFA best_sfa = bestSFA_perClass.get(class_key).getKey();
					best_sfa.featureId = all_feature_num;
					String best_sfaword_string = best_sfa.SFA_abc;

					sfa_feature.add(best_sfa);

					allFeatureList.add(new Feature(best_sfa.featureId, best_sfa.featureType, best_sfa.representLabel, best_sfa.infogain, best_sfa.threshold, best_sfa.sign));

					ArrayList<Integer> best_sfa_dis_ls_test = new ArrayList<>();
					int best_sfaword_len = best_sfaword_string.length();
					for (int j = 0; j < test_data.size(); j++) {
						TimeSeries query = test_data.get_series(j);

						short[][] sfaword_ts = sfa.transformWindowing(query);
						ArrayList<String> query_sfaword_ls = new ArrayList<>();
						for (short[] t : sfaword_ts) {
							query_sfaword_ls.add(Arrays.toString(t));
						}
						HashSet<String> ts_sfaword_set = new HashSet<>(query_sfaword_ls);
						int dis_min = Integer.MAX_VALUE;

						for (String sfaword_tmp : ts_sfaword_set) {

							int dis_tmp = 0;
							for (int i = 0; i < best_sfaword_len; i++) {
								dis_tmp+=Math.abs(best_sfaword_string.charAt(i)-sfaword_tmp.charAt(i));
							}

							if(dis_min>dis_tmp){
								dis_min = dis_tmp;
							}
						}

						best_sfa_dis_ls_test.add(dis_min);
					}
					ArrayList<Integer> tmp3 = new ArrayList<>();
					tmp3.add(all_feature_num);
					tmp3.addAll(best_sfa_dis_ls_test);
					sfa_test.add(tmp3);

                    ArrayList<Integer> best_sfa_dis_ls_validate = new ArrayList<>();
                    for (int j = 0; j < validate_data.size(); j++) {
                        TimeSeries query = validate_data.get_series(j);

                        short[][] sfaword_ts = sfa.transformWindowing(query);
                        ArrayList<String> query_sfaword_ls = new ArrayList<>();
                        for (short[] t : sfaword_ts) {
                            query_sfaword_ls.add(Arrays.toString(t));
                        }
                        HashSet<String> ts_sfaword_set = new HashSet<>(query_sfaword_ls);
                        int dis_min = Integer.MAX_VALUE;

                        for (String sfaword_tmp : ts_sfaword_set) {

                            int dis_tmp = 0;
                            for (int i = 0; i < best_sfaword_len; i++) {
                                dis_tmp+=Math.abs(best_sfaword_string.charAt(i)-sfaword_tmp.charAt(i));
                            }

                            if(dis_min>dis_tmp){
                                dis_min = dis_tmp;
                            }
                        }

                        best_sfa_dis_ls_validate.add(dis_min);
                    }
                    ArrayList<Integer> tmp4 = new ArrayList<>();
                    tmp4.add(all_feature_num);
                    tmp4.addAll(best_sfa_dis_ls_validate);
                    sfa_validate.add(tmp4);

					all_feature_num++;
				}

			}

			//遍历子节点
			if(!tmpnode.children.isEmpty()){
				for(int key: tmpnode.children.keys()){
					Node child_node = tmpnode.children.get(key);
					st.push(child_node);
				}
			}
		}
//		System.out.println(all_feature_nums);
		return forest.new SingletreeFeatureLs(allFeatureList, sfa_feature, sfa_train, sfa_validate, sfa_test, rise_feature, rise_train, rise_validate, rise_test, sax_feature, sax_train, sax_validate, sax_test);
	}

    public double tree_weighted_error(TSDataset data) throws Exception{
		double weighted_error = 0.0;
		//System.out.println("Data size" + data.size());
		for (int i = 0; i < data.size(); i++) {
			TimeSeries query = data.get_series(i);
			Integer predicted_label = predict(query, i);
			
			//TODO assert predicted_label = null;
			if (predicted_label == null) {
				throw new Exception("ERROR: possible bug detected: predicted label is null");
			}
			
			//TODO object equals or == ??
			if (!predicted_label.equals(query.getLabel())) {
				//System.out.println("predicted_label=" + predicted_label + " -- query=" + query.getLabel());
				weighted_error += query.getWeight();				
			}
		}
		weighted_error /= data.size();
		return weighted_error;
	}
	
	public Integer predict(TimeSeries query, int queryIndex) throws Exception {
		
		//System.out.println("queryIndex:"+queryIndex);
		//transform dataset using the params selected during the training phase.
		
		StringBuilder sb = new StringBuilder();
		sb.append(queryIndex + ":");
		
		
		Node node = this.root;
		
		//debug
		Node prev;
		int d = 0;
		//
		int[] labels = AppContext.getTraining_data().get_unique_classes();
		int lbl = -1;

		while(node != null && !node.is_leaf()) {
			prev = node;	//helps to debug if we store the previous node, TODO remove this later
//			System.out.println(node.splitter.getBest_splitter());
			lbl = node.splitter.predict(query, queryIndex);
//			sb.append(lbl + "-");
//			System.out.println(lbl);
//			if (node.children.get(lbl) == null) {
//				System.out.println("null child, using random choice");
//				//TODO check #class train != test
//				
//				return labels[ThreadLocalRandom.current().nextInt(labels.length)];
//			}
		
			node = node.children.get(lbl);
			if (node == null) {
				System.out.println("null node found: " + lbl);
				return lbl;
			}
			d++;
		}
		
//		if (node == null) {
//			System.out.println("null node found, returning random label ");
//			return labels[ThreadLocalRandom.current().nextInt(labels.length)];
//		}else if (node.label() == null) {
//			System.out.println("null label found, returning random label");
//			return labels[ThreadLocalRandom.current().nextInt(labels.length)];
//		}

		if (node == null) {
			System.out.println("null node found, returning exemplar label " + lbl);
			return lbl;
		}else if (node.label() == null) {
			System.out.println("null label found, returning exemplar label" + lbl);
			return lbl;
		}
		
		sb.append(">" + node.label());
		
		//System.out.println(sb.toString());
		
		return node.label();
	}	

	//TODO predict distribution
//	public int[] predict_distribution(double[] query) throws Exception {
//		Node node = this.root;
//
//		while(!node.is_leaf()) {
//			node = node.children[node.splitter.predict_by_splitter(query)];
//		}
//
//		return node.label();
//	}
//	
	public int getTreeID() {
		return tree_id;
	}

	
	//************************************** START stats -- development/debug code
	public TreeStatCollector getTreeStatCollection() {
		
		stats.collateResults(this);
		
		return stats;
	}	
	
	public int get_num_nodes() {
		if (node_counter != get_num_nodes(root)) {
			System.out.println("Error: error in node counter!");
			return -1;
		}else {
			return node_counter;
		}
	}	

	public int get_num_nodes(Node n) {
		int count = 0 ;
		
		if (n.children == null) {
			return 1;
		}
		
		for (int key : n.children.keys()) {
			count+= get_num_nodes(n.children.get(key));
		}
		
		return count+1;
	}
	
	public int get_num_leaves() {
		return get_num_leaves(root);
	}	
	
	public int get_num_leaves(Node n) {
		int count = 0 ;
		
		if (n.children == null) {
			return 1;
		}
		
		for (int key : n.children.keys()) {
			count+= get_num_leaves(n.children.get(key));
		}
		
		return count;
	}
	
	public int get_num_internal_nodes() {
		return get_num_internal_nodes(root);
	}
	
	public int get_num_internal_nodes(Node n) {
		int count = 0 ;
		
		if (n.children == null) {
			return 0;
		}
		
		for (int key : n.children.keys()) {
			count+= get_num_internal_nodes(n.children.get(key));
		}
		
		return count+1;
	}
	
	public int get_height() {
		return get_height(root);
	}
	
	public int get_height(Node n) {
		int max_depth = 0;
		
		if (n.children == null) {
			return 0;
		}
		
		for (int key : n.children.keys()) {
			max_depth = Math.max(max_depth, get_height(n.children.get(key)));
		}

		return max_depth+1;
	}
	
	public int get_min_depth(Node n) {
		int max_depth = 0;
		
		if (n.children == null) {
			return 0;
		}
		
		for (int key : n.children.keys()) {
			max_depth = Math.min(max_depth, get_height(n.children.get(key)));
			
		}
		
		return max_depth+1;
	}
	
//	public double get_weighted_depth() {
//		return printTreeComplexity(root, 0, root.data.size());
//	}
//	
//	// high deep and unbalanced
//	// low is shallow and balanced?
//	public double printTreeComplexity(Node n, int depth, int root_size) {
//		double ratio = 0;
//		
//		if (n.is_leaf) {
//			double r = (double)n.data.size()/root_size * (double)depth;
////			System.out.format("%d: %d/%d*%d/%d + %f + ", n.label, 
////					n.data.size(),root_size, depth, max_depth, r);
//			
//			return r;
//		}
//		
//		for (int i = 0; i < n.children.length; i++) {
//			ratio += printTreeComplexity(n.children[i], depth+1, root_size);
//		}
//		
//		return ratio;
//	}		
	
	
	//**************************** END stats -- development/debug code
	
	
	
	
	
	
	
	public class Node{
	
		protected transient Node parent;	//dont need this, but it helps to debug
		public transient ProximityTree tree;		
		
		protected int node_id;
		protected int node_depth = 0;
		protected int node_num = 0;

		protected boolean is_leaf = false;
		protected Integer label;

//		protected transient Dataset data;	
		//class distribution of data passed to this node during the training phase
		protected TIntIntMap class_distribution; 
		protected String class_distribution_str = ""; //class distribution as a string, just for printing and debugging
		protected TIntObjectMap<Node> children;
		protected SplitEvaluator splitter;

		double tree_splitter_time = 0;

		public TIntObjectMap<Node> getChildren() {
			return children;
		}

		public SplitEvaluator getSplitter() {
			return splitter;
		}

		public Node(Node parent, Integer label, int node_id, ProximityTree tree) {
			this.parent = parent;
//			this.data = new ListDataset();
			this.node_id = node_id;
			this.tree = tree;
			
			if (parent != null) {
				node_depth = parent.node_depth + 1;
			}
		}
		
		public boolean is_leaf() {
			return this.is_leaf;
		}
		
		public Integer label() {
			return this.label;
		}	
		
		public TIntObjectMap<Node> get_children() {
			return this.children;
		}		
		
//		public Dataset get_data() {
//			return this.data;
//		}		
		
		public String toString() {
			return "d: " + class_distribution_str;// + this.data.toString();
		}		

		
//		public void train(Dataset data) throws Exception {
//			this.data = data;
//			this.train();
//		}		
		
		public void train(TSDataset data) throws Exception {
//			out.write(this.node_depth + ":   " + (this.parent == null ? "r" : this.parent.node_id)  +"->"+ this.node_id +":"+ data.toString() + "\n");
//			System.out.println(this.node_depth + ":   " + (this.parent == null ? "r" : this.parent.node_id)  +"->"+ this.node_id +":"+ data.toString());
			this.node_num ++;
			
			//Debugging check
			if (data == null) {
//				throw new Exception("possible bug: empty node found");
//				this.label = Util.majority_class(data);
				this.is_leaf = true;
				System.out.println("node data == null, returning");
				return;				
			}
			
			this.class_distribution = data.get_class_map(); //TODO do we need to clone it? nope
			
			if (data.size() == 0) {
				this.is_leaf = true;
				System.out.println("node data.size == 0, returning");
				return;			
			}
			
			if (data.gini() == 0) {
				this.label = data.get_class(0);	//works if pure
				this.is_leaf = true;
				return;
			}
			
			// Minimum leaf size
			if (data.size() <= AppContext.min_leaf_size) {
				this.label = data.get_majority_class();	//-- choose the majority class at the node
				this.is_leaf = true;
				return;
			}

//			this.splitter = SplitterChooser.get_random_splitter(this);
			this.splitter = new SplitEvaluator(this);
				
			TIntObjectMap<TSDataset> best_splits = splitter.train(data);
			tree_splitter_time += splitter.splitter_time;
			
			//TODO refactor
			if (best_splits == null || has_empty_split(best_splits)) {
				//stop training and create a new leaf
//				throw new Exception("Empty leaf found");
				this.label = Util.majority_class(data);
				this.is_leaf = true;
//				System.out.println("Empty split found...returning a leaf: " + this.class_distribution + " leaf_label: " + this.label);
				
				return;
			}
			
			this.children = new TIntObjectHashMap<Node>(best_splits.size());
			
//			System.out.println(Arrays.toString(best_splits.keys()));
			
			for (int key : best_splits.keys()) {
				this.children.put(key, new Node(this, key, ++tree.node_counter, tree));
				this.children.get(key).class_distribution_str = best_splits.get(key).get_class_map().toString(); //TODO debug only mem- leak
			}
			
			for (int key : best_splits.keys()) {
				this.children.get(key).train(best_splits.get(key));
			}

		}

	}
	
	public boolean has_empty_split(TIntObjectMap<TSDataset> splits) throws Exception {
		
		for (int key : splits.keys()) {
			if (splits.get(key) == null || splits.get(key).size() == 0) {
				return true;
			}
		}

		return false;
	}
	
}
