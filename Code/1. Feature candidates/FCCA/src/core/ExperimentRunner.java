package core;

import java.io.File;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

import org.apache.commons.lang3.StringUtils;

import datasets.TSDataset;
import trees.ProximityForest;
import util.PrintUtilities;


public class ExperimentRunner {
	
	TSDataset train_data;
	TSDataset test_data;
	TSDataset validate_data;
//	private static String csvSeparatpr = ",";
	private static String csvSeparatpr = "\t";//读取ucr原始数据
	
	private MultiThreadedTasks parallel_tasks;

	public ExperimentRunner(){
		
		parallel_tasks = new MultiThreadedTasks();

	}
	
	public void run() throws Exception {
		
		//read data files
		//we assume no header in the csv files, and that class label is in the first column, modify if necessary
		TSDataset train_data_original = 
				CSVReader.readCSVToTSDataset(AppContext.training_file, AppContext.csv_has_header, 
						AppContext.target_column_is_first, csvSeparatpr, AppContext.verbosity);
		TSDataset test_data_original = 
				CSVReader.readCSVToTSDataset(AppContext.testing_file, AppContext.csv_has_header, 
						AppContext.target_column_is_first, csvSeparatpr, AppContext.verbosity);

		TSDataset validate_data_original =
				CSVReader.readCSVToTSDataset(AppContext.validate_file, AppContext.csv_has_header,
						AppContext.target_column_is_first, csvSeparatpr, AppContext.verbosity);

		train_data = train_data_original; //train_data_original.reorder_class_labels(null);
		test_data = test_data_original; //test_data_original.reorder_class_labels(train_data._get_initial_class_labels());
		validate_data = validate_data_original;

		AppContext.setTraining_data(train_data);
		AppContext.setTesting_data(test_data);
		AppContext.setValidate_data(validate_data);
		AppContext.updateClassDistribution(train_data, test_data);
				
		//allow garbage collector to reclaim this memory, since we have made copies with reordered class labels
		train_data_original = null;
		test_data_original = null;
		System.gc();

		//setup environment
		File training_file = new File(AppContext.training_file);
		String datasetName = training_file.getName().replaceAll("_TRAIN.tsv", "");	//this is just some quick fix for UCR datasets
		AppContext.setDatasetName(datasetName);

		if (AppContext.verbosity > 0) {
			PrintUtilities.printDatasetInfo();
		}

		//if we need to shuffle
		if (AppContext.shuffle_dataset) {
			System.out.println("Shuffling the training set...");
			train_data.shuffle(AppContext.rand_seed);	//NOTE seed
			test_data.shuffle(AppContext.rand_seed);
		}
		
		String outputFilePrefix = createOutPutFile(datasetName);	//giving same prefix for all repeats so thats easier to group using pandas
		AppContext.currentOutputFilePrefix = outputFilePrefix;
				
		for (int i = 0; i < AppContext.num_repeats; i++) {
			
			if (AppContext.verbosity > 0) {
				System.out.println("======================================== Repetition No: " + (i+1) + " (" +datasetName+ ") ========================================");
				
				if (AppContext.verbosity > 1) {
					System.out.println("Threads: MaxPool=" + parallel_tasks.getThreadPool().getMaximumPoolSize() 
							+ ", Active: " + parallel_tasks.getThreadPool().getActiveCount()
							+ ", Total: " + Thread.activeCount());					
				}


			}else if (AppContext.verbosity == 0 && i == 0){
				System.out.println("#,Repetition, Dataset, Accuracy, TrainTime(ms), TestTime(ms), AvgDepthPerTree");
			}

			File rootDir = new File(AppContext.feature_save_file);
			if(!rootDir.exists()){
				rootDir.mkdir();
			}

			String rootResult = AppContext.feature_save_file+"\\";
			AppContext.resultRootPath = rootResult;

			//create model
			ProximityForest forest = new ProximityForest(i, parallel_tasks);
			
			//train model
			forest.train_parallel(train_data);
			System.out.println("============ train finished ============");

			forest.getForestFeature_parallel(train_data, test_data, validate_data);
			System.out.println("========== getFeature finished ===========");
			
			if (AppContext.verbosity > 1) {
				System.out.println("Threads: MaxPool=" + parallel_tasks.getThreadPool().getMaximumPoolSize() 
						+ ", Active: " + parallel_tasks.getThreadPool().getActiveCount()
						+ ", Total: " + Thread.activeCount());				
			}
			
			if (AppContext.garbage_collect_after_each_repetition) {
				System.gc();
			}

		}
		
		parallel_tasks.getExecutor().shutdown();
		parallel_tasks.getExecutor().awaitTermination(Long.MAX_VALUE, TimeUnit.DAYS);	//just wait forever
				
	
	}

	private String createOutPutFile(String datasetName) throws IOException {
		AppContext.experiment_timestamp = LocalDateTime.now();
		String timestamp = AppContext.experiment_timestamp.format(DateTimeFormatter.ofPattern(AppContext.FILENAME_TIMESTAMP_FORMAT));			
		
		String fileName = timestamp + "_" + datasetName;
				
		return fileName;
	}

}
