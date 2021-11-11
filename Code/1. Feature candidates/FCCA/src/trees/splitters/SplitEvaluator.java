package trees.splitters;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import core.AppContext;
import core.AppContext.FeatureSelectionMethod;
import core.AppContext.RifFilters;
import core.AppContext.SplitterType;
import core.TreeStatCollector;
import datasets.TSDataset;
import datasets.TimeSeries;
import gnu.trove.map.TIntObjectMap;
import trees.ProximityTree;
import util.Sampler;

public class SplitEvaluator{

	List<NodeSplitter> splitters; //changed array to a list, to make shuffling easier
	ProximityTree.Node node;
	NodeSplitter best_splitter;
	
	List<NodeSplitter> best_splitters;	//store the best ones to pick a random splitter if more than one are equal

	public Double splitter_time = 0.0;
	
	public SplitEvaluator(ProximityTree.Node node) throws Exception {
		this.node = node;	
		
		splitters = new ArrayList<NodeSplitter>(AppContext.num_actual_splitters_needed);
		best_splitters = new ArrayList<NodeSplitter>(3);	//initial capacity is set to a small number, its unlikely that splitters would have same gini very often
		
	}

	public NodeSplitter getBest_splitter() {
		return best_splitter;
	}

	// 找到调用该函数的节点上的最佳分类
	public TIntObjectMap<TSDataset> train(TSDataset data) throws Exception {
		long seStartTime = System.nanoTime();

		splitter_time = 0.0;

		double weighted_gini = Double.POSITIVE_INFINITY;
		double best_weighted_gini = Double.POSITIVE_INFINITY;
		
		TIntObjectMap<TSDataset> splits = null;
		TIntObjectMap<TSDataset> best_split = null;
		
		double parent_size;
		if (!AppContext.boosting) {
			parent_size = (double) data.size();	
		}
		else {
			parent_size = data.get_sum_weight();
		}

		chooseSplitters(); // 获取所有提出的splitter（参数中指定的，例如该节点要求有3个ee，4个boss 这种）
		
		//shuffle splitters here, if not using random tie break	--NOTE not necessary if tie breaking randomly
//		Collections.shuffle(splitters);
		
		//find the correct indices
		int node_size = data.size();
		int[] node_indices = new int[node_size];
		
		TSDataset all_train_data = this.node.tree.train_data;
		int total_train_size = all_train_data.size();
		
		//TODO HIGH PRIORITY -- increases runtime a lot, TEST can be avoided if using indices at all places
		long st2 = System.nanoTime();
		for (int i = 0; i < total_train_size; i++) {
			for (int j = 0; j < node_size; j++) {
				if (all_train_data.get_series(i) == data.get_series(j)) {
					node_indices[j] = i;
				}
			}
		}

		splitter_time += ((System.nanoTime() - seStartTime)/AppContext.NANOSECtoMILISEC);
		this.node.tree.stats.data_fetch_time += (System.nanoTime() - st2);
		
		int n = splitters.size();
//		System.out.println("splitters.size():"+n);
		TreeStatCollector stats = this.node.tree.stats;
		RIFSplitter temp_rif;

		double node_sax_time = 0.0;
		double node_sfa_time = 0.0;
		
		for (int i = 0; i < n; i++) {

//			File featureDir = new File(nodePath +"feature_"+ i);
//			if(!featureDir.exists()){
//				featureDir.mkdir();
//			}
//			String featurePath = nodePath +"feature_"+ i + "\\";

			TSDataset sample;
			//n' = sample(2,per_class,1.0)
			
			if (AppContext.gini_approx) {
				sample = sample(data, AppContext.approx_gini_min, AppContext.approx_gini_min_per_class);
			}else {
				sample = data; //save time by not copying all the data, if we do gini on 100% data
			}			
			
			//collect stats 
			
			NodeSplitter currentSplitter = splitters.get(i);
			
			long startTime = System.nanoTime();
			
			if (currentSplitter instanceof BossSplitter) {
				//splitter.train(n)
				splits = currentSplitter.train(data, node_indices);
				//splitter.test(n')
//				splits = splitters[i].split(data, null);
				stats.boss_time += (System.nanoTime() - startTime);
				node_sfa_time += ((BossSplitter) currentSplitter).train_node_time;

			}else if(currentSplitter instanceof SaxSplitter){

				splits = currentSplitter.train(data, node_indices);
				stats.sax_time += (System.nanoTime() - startTime);
				node_sax_time += ((SaxSplitter) currentSplitter).train_node_time;

			} else if (currentSplitter instanceof RIFSplitter) {
				splits = currentSplitter.train(data, node_indices);
				
			}else {
				splits = currentSplitter.train(data, node_indices);
			}

			if(splits==null) continue;
			
			weighted_gini = weighted_gini(parent_size, splits);


//TODO if equal take a random choice? or is it fine without it
			if (weighted_gini <  best_weighted_gini) {
				best_weighted_gini = weighted_gini;
				best_split = splits;
				best_splitter = currentSplitter;
				best_splitters.clear();
				best_splitters.add(currentSplitter);
			}else if (weighted_gini ==  best_weighted_gini) {	//NOTE if we enable this then we need update best_split again -expensive
				best_splitters.add(splitters.get(i));
			}
		}

		double node_rise_time = stats.rif_time/AppContext.NANOSECtoMILISEC;

		splitter_time += (node_sax_time + node_sfa_time + node_rise_time);

		//failed to find any valid split point
		if (best_splitters.size() == 0) {
			return null;
		}else if (best_splitters.size() == 1) {
			best_splitter = best_splitters.get(0);	//then use stored best split
		}else { //if we have more than one splitter with equal best gini
			int r =  ThreadLocalRandom.current().nextInt(best_splitters.size());
			best_splitter = best_splitters.get(r);	
			// then we need to find the best split again, can't resuse best split we stored before.
			best_split = best_splitter.split(data, null);
			
			if (AppContext.verbosity > 4) {
//				System.out.println("best_splitters.size() == " + best_splitters.size());
			}
		}
		
		//split the whole dataset using the (approximately) best splitter
//		best_split =  best_splitter.split(data, null); //TODO check
		
		//allow gc to deallocate unneeded memory
//		for (int j = 0; j < splitters.length; j++) {
//			if (splitters[j] != best_splitter) {
//				splitters[j] = null;
//			}
//		}
		splitters = null;
		best_splitters.clear();//clear the memory

		storeStatsForBestSplitter();
		
		this.node.tree.stats.split_evaluator_train_time += (System.nanoTime() - seStartTime);
		return best_split;
	}
	
	public String toString() {
		if (best_splitter == null){
			return "untrained";
		}else {
			return best_splitter.toString();
		}
	}
	
	private void chooseSplitters() throws Exception {
		
		int n = AppContext.num_splitters_per_node;

		int total_added = 0;

		if(AppContext.sax_enabled){
			for (int i = 0; i < AppContext.sax_splitters_per_node; i++) {

				splitters.add(new SaxSplitter(node));
				total_added++;
			}
		}

		if (AppContext.boss_enabled) {
			for (int i = 0; i < AppContext.boss_splitters_per_node; i++) {

				splitters.add(new BossSplitter(node));
				total_added++;
			}
		}

		if (AppContext.rif_enabled) {

			int rif_total = AppContext.num_actual_rif_splitters_needed_per_type * 2;
			int reminder_gini = AppContext.rif_splitters_per_node - (AppContext.rif_m * rif_total);
			int m;

			for (int i = 0; i < rif_total; i++) {

				if (AppContext.rif_components == RifFilters.ACF_PACF_ARMA_PS_separately) {
					RIFSplitter splitter;

					//TODO quick fix to make sure that rif_splitters_per_node of ginis are actually calculated
					if (reminder_gini > 0) {
						m = AppContext.rif_m + 1;
						reminder_gini--;
					}else {
						m = AppContext.rif_m;
					}

					//divide appox equally (best if divisible by 4)
					if (i % 4 == 0) {
						RIFSplitter sp = new RIFSplitter(node, RifFilters.ACF, m);
						splitters.add(sp);
					}else if (i % 4 == 1) {
						RIFSplitter sp = new RIFSplitter(node, RifFilters.PACF, m);
						splitters.add(sp);
					}else if (i % 4 == 2) {
						RIFSplitter sp = new RIFSplitter(node, RifFilters.ARMA, m);
						splitters.add(sp);
					}else if (i % 4 == 3) {
						RIFSplitter sp = new RIFSplitter(node, RifFilters.PS, m);
						splitters.add(sp);
					}

				}
				else if(AppContext.rif_components == RifFilters.ACF_PS_separately){
					if (reminder_gini > 0) {
						m = AppContext.rif_m + 1;
						reminder_gini--;
					}else {
						m = AppContext.rif_m;
					}

					//divide appox equally (best if divisible by 4)
					if (i % 2 == 0) {
						RIFSplitter sp = new RIFSplitter(node, RifFilters.ACF, m);
						splitters.add(sp);
					}else if (i % 2 == 1) {
						RIFSplitter sp = new RIFSplitter(node, RifFilters.PS, m);
						splitters.add(sp);
					}
				}
				else {
					splitters.add(new RIFSplitter(node, AppContext.rif_components));
				}

				total_added++;
			}
		}
	}
	
	private static NodeSplitter get_random_splitter(ProximityTree.Node node) throws Exception {
		
		NodeSplitter splitter = null;
		
		int r = ThreadLocalRandom.current().nextInt(AppContext.enabled_splitters.length);
//		int r = AppContext.getRand().nextInt(AppContext.enabled_splitters.length);
		SplitterType selection = AppContext.enabled_splitters[r];
		
		//use split constraints to set up limits to the splitting condition 
		
		switch(selection) {
			case BossSplitter:
//				System.out.println("choosing BossSplitterPerTree");
				splitter = new BossSplitter(node);
				
//				splitter = new BossSplitterPerNode(node);
				break;
			case RIFSplitter:
				splitter = new RIFSplitter(node, AppContext.rif_components);
				break;
			default:
				throw new Exception("Splitter type not supported");
		}
		
		return splitter;
	}
	
	private void storeStatsForBestSplitter() {
		if (best_splitter instanceof BossSplitter) {
			this.node.tree.stats.boss_win++;
		}else if(best_splitter instanceof SaxSplitter){
			this.node.tree.stats.sax_win++;
		}else if (best_splitter instanceof RIFSplitter) {
			this.node.tree.stats.rif_win++;
			
			RIFSplitter rif = (RIFSplitter) best_splitter;
			if (rif.filter_type.equals(RifFilters.ACF)) {
				this.node.tree.stats.rif_acf_win++;
			}else if (rif.filter_type.equals(RifFilters.PACF)) {
				this.node.tree.stats.rif_pacf_win++;
			}else if (rif.filter_type.equals(RifFilters.ARMA)) {
				this.node.tree.stats.rif_arma_win++;
			}else if (rif.filter_type.equals(RifFilters.PS)) {
				this.node.tree.stats.rif_ps_win++;
			}else if (rif.filter_type.equals(RifFilters.DFT)) {
				this.node.tree.stats.rif_dft_win++;
			}
		}
	}
	
	
	public static boolean has_empty_split(TIntObjectMap<TSDataset> splits) throws Exception {
		
		for (int key : splits.keys()) {
			if (splits.get(key) == null || splits.get(key).size() == 0) {
				return true;
			}
		}

		return false;
	}
	
	
	
//	//takes the max(gini_min * #class_root, gini_approx * n)
//	private Dataset sample(Dataset data, int approx_gini_min, boolean approx_gini_min_per_class, double approx_gini_percent) {
//		Dataset sample;
//		int sample_size = (int) (approx_gini_percent * data.size());
//		int min_sample_size;
//
//		//use number of classes in the root
//		if (approx_gini_min_per_class) {
//			min_sample_size = approx_gini_min * AppContext.getTraining_data().get_num_classes();
//			if (sample_size  < min_sample_size) {
//				sample_size = min_sample_size;
//			}
//		}else {
//			if (sample_size < approx_gini_min) {
//				sample_size = approx_gini_min;
//			}		
//		}
//		
//		sample = Sampler.uniform_sample(data, sample_size);
//		
//		return sample;
//	}

	//takes the min(gini_min * #class_root, n)
	private TSDataset sample(TSDataset data, int approx_gini_min, boolean approx_gini_min_per_class) {
		TSDataset sample;
		int sample_size;

		//use number of classes in the root
		if (approx_gini_min_per_class) {
			sample_size = Math.min(approx_gini_min * AppContext.getTraining_data().get_num_classes(), data.size());
		}else {
			sample_size = Math.min(approx_gini_min, data.size());	
		}
		
		sample = Sampler.uniform_sample(data, sample_size);
		
		return sample;
	}
	
	public int predict(TimeSeries query, int queryIndex) throws Exception {
		return best_splitter.predict(query, queryIndex);
	}


	
	
	
	public double weighted_gini(double parent_size, TIntObjectMap<TSDataset> splits) {
		double wgini = 0.0;
		double gini;
		double split_size = 0;
		
		if (splits == null) {
			return Double.POSITIVE_INFINITY;
		}
		
		if (!AppContext.boosting) {
			for (int key : splits.keys()) {
				if (splits.get(key) == null) {	//NOTE
					gini = 1;
					split_size = 0;
				}else {
					gini = splits.get(key).gini();
					split_size = (double) splits.get(key).size();
				}
				wgini = wgini + (split_size / parent_size) * gini;
			}
		} else {
			for (int key : splits.keys()) {
				if (splits.get(key) == null) {	//NOTE
					gini = 1;
					split_size = 0;
				}else {
					gini = splits.get(key).weighted_gini();
					split_size = splits.get(key).get_sum_weight();
				}
				wgini = wgini + (split_size / parent_size) * gini;
			}
		}

		return wgini;
	}


}
