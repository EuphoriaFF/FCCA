package dev;

import java.util.*;

import com.carrotsearch.hppc.cursors.IntIntCursor;

import core.AppContext;
import datasets.BossDataset;
import datasets.BossDataset.BossParams;
import datasets.TSDataset;
import datasets.TimeSeries;
import transforms.BOSS;
import transforms.SFA;
import transforms.SFA.HistogramType;
import transforms.SFA.Words;
import trees.splitters.BossSplitter;
import transforms.BOSS.BagOfPattern;

public class BossTransformContainer implements TransformContainer {

	int alphabetSize = 4;
	int minWindowLength = 10;
	int maxWindowLength = 0;	//if 0 then assume maxWindowLength == length of series
	int windowLengthStepSize = 1;
	
	boolean use_lower_bound = true; //TODO not implemented?? CHECK
	
	boolean[] NORMALIZATION = new boolean[] { false };

	public HashMap<String, SFA> sfa_transforms;	//change String keys to Int keys
	public HashMap<String, BossDataset> boss_datasets;	//change String keys to Int keys
	public List<BossDataset.BossParams> boss_params;

	public HashMap<String, ArrayList<ArrayList<String>>> datasets_to_sfa;
	
	int num_transforms;
	public static HashMap<Integer, short[]> wordtolong;
	
	SplittableRandom rand = new SplittableRandom();

	public BossTransformContainer(int boss_num_transforms) {
		sfa_transforms = new HashMap<String, SFA>();
		boss_datasets = new HashMap<String, BossDataset>(boss_num_transforms);
		boss_params = new ArrayList<>(boss_num_transforms);	//TODO not necessary
		num_transforms = boss_num_transforms;

		datasets_to_sfa = new HashMap<>(boss_num_transforms);

		wordtolong = new HashMap<>();
		
	}
	
	
	
	
	//hacky implementation TODO refactor later
	private ArrayList<Integer> selectRandomParams(TSDataset train) {
		
		if (maxWindowLength == 0 || maxWindowLength > train.length()) {
			maxWindowLength = train.length();
		}
		
		List<BossDataset.BossParams> all_params = new ArrayList<>();
		
		int[] wordLengths = getBossWordLengths();
		
		for (boolean normMean : NORMALIZATION) {
			for (int w = minWindowLength; w < maxWindowLength; w+=windowLengthStepSize) {
				for (int l : wordLengths) {
					all_params.add(new BossParams(normMean, w, l, alphabetSize));
					
				}
			}
		}
		
		//pick num_transform items randomly
		
		ArrayList<Integer> indices = new ArrayList<>(all_params.size());
		for (int i = 0; i < all_params.size(); i++) {
			indices.add(i);
		}
		
		Collections.shuffle(indices);
		
		//TODO if #all possible params is < num_transformation, update max limit
		if (num_transforms > all_params.size()) {
			num_transforms = all_params.size();
			if (AppContext.verbosity > 1) {
				System.out.println("INFO: boss_num_transformation has been updated to: " + num_transforms + " (#all possible params with given settings ("+all_params.size()+") < the given num_transformations)");
			}
		}
		for (int i = 0; i < num_transforms; i++) {
			//clone
			BossParams temp = all_params.get(indices.get(i));
			
			boss_params.add(new BossParams(temp.normMean, temp.window_len, temp.word_len, temp.alphabet_size));
		};
		
		all_params = null; //let GC free this memory
		
		return indices;
	}
	

	@Override
	public void fit(TSDataset train) {
//		System.out.println("computing train transformations: " + num_transforms);
		TimeSeries[] samples = train._get_internal_list().toArray(new TimeSeries[] {});

		selectRandomParams(train);
		
		for (int param = 0; param < boss_params.size(); param++) {
			BossParams temp = boss_params.get(param);
			
			SFA sfa = new SFA(HistogramType.EQUI_DEPTH);
			//为了建立bin,便于后面分桶操作
			sfa.fitWindowing(samples, temp.window_len, temp.word_len, temp.alphabet_size, temp.normMean, use_lower_bound);

			BossDataset boss_dataset = transform_dataset_using_sfa(train,sfa, temp, samples);
			
			String key =  temp.toString();
			
			sfa_transforms.put(key, sfa);
			boss_datasets.put(key, boss_dataset);


			
		}
	}




	@Override
	public TSDataset transform(TSDataset test) {
		
		System.out.println("computing test transformations: " + num_transforms);
		TimeSeries[] samples = test._get_internal_list().toArray(new TimeSeries[] {});
		
		for (int param = 0; param < boss_params.size(); param++) {
			BossParams temp = boss_params.get(param);
			SFA sfa = sfa_transforms.get(temp.toString());
			BossDataset boss_dataset = transform_dataset_using_sfa(test,sfa, temp, samples);
			
			String key =  temp.toString();
			key += "_test";
			
			sfa_transforms.put(key, sfa);
			boss_datasets.put(key, boss_dataset);
		}
		return null;
	}
	
	
	//samples array is a quick fix, will remove later
	public BossDataset transform_dataset_using_sfa(TSDataset dataset, SFA sfa, BossParams params, TimeSeries[] samples) {
		
		final int[][] words = new int[samples.length][];

		ArrayList<ArrayList<String>> dataset_sfawordLs = new ArrayList<>();
		
        for (int i = 0; i < samples.length; i++) {
              short[][] sfaWords = sfa.transformWindowing(samples[i]);// 第一维; sfa-word的个数 第二维：每个SFA-word的表达

			ArrayList<String> ts_sfawordls = new ArrayList<>();

              words[i] = new int[sfaWords.length]; // 第i个时间序列一共有j个SFA-word
              for (int j = 0; j < sfaWords.length; j++) {
              	ts_sfawordls.add(Arrays.toString(sfaWords[j]));
                words[i][j] = (int) Words.createWord(sfaWords[j], sfa.wordLength, (byte) Words.binlog(sfa.alphabetSize));
              }
              dataset_sfawordLs.add(ts_sfawordls);
          }

		BagOfPattern[] histograms = BossSplitter.createBagOfPattern(words, samples, sfa.wordLength, sfa.alphabetSize,dataset_sfawordLs);

		//TODO change params
		BossDataset boss_dataset = new BossDataset(dataset, histograms, params);
		datasets_to_sfa.put(params.toString(), dataset_sfawordLs);
		
		return boss_dataset;
	}
	
	
	public BagOfPattern transform_series_using_sfa(TimeSeries series, SFA sfa, int queryIndex) {

		final int[] words;

		short[][] sfaWords = sfa.transformWindowing(series);
		ArrayList<String> ts_sfaword_ls = new ArrayList<>();

		words = new int[sfaWords.length];
		for (int j = 0; j < sfaWords.length; j++) {
			ts_sfaword_ls.add(Arrays.toString(sfaWords[j]));
			words[j] = (int) Words.createWord(sfaWords[j], sfa.wordLength, (byte) Words.binlog(sfa.alphabetSize));
		}

		BagOfPattern histogram = BossSplitter.createBagOfPattern(words, series, sfa.wordLength,sfa.alphabetSize, queryIndex,ts_sfaword_ls);

		return histogram;
	}
	
	public static int[] getBossWordLengths() {
		//TODO
//		for (int l = minWordLength; l <= maxWordLength; l += wordLengthStepSize) {
//
//		}
	
		return new int[]{6,8,10,12,14,16};	
	}

}
