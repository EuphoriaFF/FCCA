package core;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import datasets.TSDataset;
import datasets.TimeSeries;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.hash.TIntIntHashMap;
import trees.ProximityForest;
import trees.ProximityForest.Predictions;
import trees.ProximityTree;
import util.PrintUtilities;

public class MultiThreadedTasks {

	private ExecutorService executor;
	private int pool_size;

	public MultiThreadedTasks() {

		if (AppContext.num_threads == 0) { // auto
			AppContext.num_threads = Runtime.getRuntime().availableProcessors();
		}

		setExecutor(Executors.newFixedThreadPool(AppContext.num_threads));

		// this is important for slurm jobs because
		// Runtime.getRuntime().availableProcessors() does not equal SBATCH argument
		// cpus-per-task
		if (executor instanceof ThreadPoolExecutor) {
			pool_size = ((ThreadPoolExecutor) executor).getMaximumPoolSize();
		}

//		System.out.println("Using " + pool_size + " threads (CPUs=" + Runtime.getRuntime().availableProcessors() + ")");

	}

//	public ExecutorService createExecutor(int num_threads) {
//		return Executors.newFixedThreadPool(num_threads);
//	}

	public ExecutorService getExecutor() {
		return executor;
	}

	public void setExecutor(ExecutorService executor) {
		this.executor = executor;
	}

	public ThreadPoolExecutor getThreadPool() {

		if (executor instanceof ThreadPoolExecutor) {
			return ((ThreadPoolExecutor) executor);
		} else {
			return null;
		}
	}

	public class TrainingTask implements Callable<Integer> {
		ProximityTree[] trees;
		TSDataset train;
		int start;
		int end;

		public TrainingTask(ProximityTree[] trees, int start, int end, TSDataset train) {
			this.trees = trees;
			this.start = start;
			this.end = end;
			this.train = train;
		}

		@Override
		public Integer call() throws Exception {
			
			for (int k = start; k < end; k++) {

				trees[k].train(train, k);

				if (AppContext.verbosity > 0) {
					//System.out.print(k + ".");
					if (AppContext.verbosity > 2) {
						PrintUtilities.printMemoryUsage(true);

					}
				}
			}

			return null;
		}

	}

	public class TestingPerModelTask implements Callable<Integer> {
		ProximityForest forest;
		TSDataset test_data;
		List<Integer> model_indices;
		Predictions[] predictions; // must be initialized by the caller, this array is shared by multiple threads,
									// size must equal to no. of models

		public TestingPerModelTask(ProximityForest ensemble, List<Integer> base_model_indices, TSDataset test,
				Predictions[] predictions) {
			this.forest = ensemble;
			this.model_indices = base_model_indices;
			this.test_data = test;
			this.predictions = predictions;
		}

		@Override
		public Integer call() throws Exception {

			for (int i = 0; i < model_indices.size(); i++) {
				int index = model_indices.get(i);
				predictions[index] = forest.getTree(index).predict(test_data);
				if (AppContext.verbosity > 0) {
					System.out.print(".");
				}

			}

			return null;
		}

	}

	public class GetfeaturePerModelTask implements Callable<Integer> {

		ProximityForest forest;
		TSDataset train_data;
		TSDataset test_data;
		TSDataset validate_data;
		List<Integer> model_indices;
		ProximityForest.SingletreeFeatureLs[] singlereeFeatureLs;

		public GetfeaturePerModelTask(ProximityForest forest, TSDataset train_data, TSDataset test_data, List<Integer> model_indices, ProximityForest.SingletreeFeatureLs[] singlereeFeatureLs, TSDataset validate_data) {
			this.forest = forest;
			this.train_data = train_data;
			this.test_data = test_data;
			this.validate_data = validate_data;
			this.model_indices = model_indices;
			this.singlereeFeatureLs = singlereeFeatureLs;
		}

		@Override
		public Integer call() throws Exception {

			for (int i = 0; i < model_indices.size(); i++){
				int index = model_indices.get(i);
				singlereeFeatureLs[index] = forest.getTree(index).get_singletree_feature_distance_parallel(train_data, test_data,validate_data,0);

			}

			return null;
		}
	}

}
