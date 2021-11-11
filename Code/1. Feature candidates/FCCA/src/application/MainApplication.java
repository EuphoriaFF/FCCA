package application;

import java.net.InetAddress;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import core.AppContext;
import core.ExperimentRunner;
import core.AppContext.FeatureSelectionMethod;
import core.AppContext.ParamSelection;
import core.AppContext.RifFilters;
import core.AppContext.SplitMethod;
import core.AppContext.SplitterType;
import core.AppContext.TransformLevel;
import util.Util;
import util.PrintUtilities;

public class MainApplication {
	public static boolean use_ucr = false;

	public static final String[] dev_args1 = new String[]{
			"-train=.\\data\\Coffee\\Coffee_TRAIN.tsv",
			"-test=.\\data\\Coffee\\Coffee_VALIDATE.tsv",
			"-validate=.\\data\\Coffee\\Coffee_Test.tsv",
			"-feature_file=.\\result\\Coffee",
			"-repeats=1",
			"-trees=100",
			"-s=sax:100,boss:100,rise:100",
			"-export=1",
			"-verbosity=0",
	};
	

	public static String[] binary_usr_datasets1 = {"ItalyPowerDemand"};

	public static void main(String[] args) {
		try {
			for(int dataset_name_id=0; dataset_name_id<binary_usr_datasets1.length; dataset_name_id++){

				AppContext.version = "1.0.0.beta";
				AppContext.version_tag = "v010000.beta first release - code cleaning in progress";
				AppContext.release_notes = AppContext.release_notes + String.join(
						"v010000.beta first release - code cleaning in progress; hence beta",
						"build_at=3/7/2019");

				String dataset_name = binary_usr_datasets1[dataset_name_id];

				args = dev_args1;
				args[0] = "-train=data\\"+dataset_name+"\\"+dataset_name+"_TRAIN.tsv";
				args[1] = "-test=data\\"+dataset_name+"\\"+dataset_name+"_TEST.tsv";
				args[2] = "-validate=data\\"+dataset_name+"\\"+dataset_name+"_VALIDATE.tsv";
				args[3] = "-feature_file=result\\"+dataset_name;

				AppContext.cmd_args = args; //store

				//sysinfo
				AppContext.host_name = InetAddress.getLocalHost().getHostName();
				AppContext.application_start_timestamp = LocalDateTime.now();
				AppContext.application_start_timestamp_formatted = AppContext.application_start_timestamp.format(DateTimeFormatter.ofPattern(AppContext.FILENAME_TIMESTAMP_FORMAT));

				parseCMD(args);

				AppContext.current_experiment_id = java.util.UUID.randomUUID().toString().substring(0, 8);
				//these unique ids helps to keep the output folder structure clean
				AppContext.repeat_guids = new ArrayList<>(AppContext.num_repeats);
				for (int i = 0; i < AppContext.num_repeats; i++) {
					AppContext.repeat_guids.add(java.util.UUID.randomUUID().toString().substring(0, 8));
				}

				//experiment
				if (AppContext.warmup_java) {
					Util.warmUpJavaRuntime();
				}

				//experiment
				if (AppContext.boss_param_selection == ParamSelection.PreLoadedSet) {
					AppContext.boss_preloaded_params = AppContext.loadBossParams();
				}

				if (AppContext.verbosity > 0) {
					System.out.println("Version: " + AppContext.version + " (" + AppContext.version_tag + ")");
					PrintUtilities.printConfiguration();
//				System.out.println();
				}

				ExperimentRunner experiment = new ExperimentRunner();
				experiment.run();

			}

		}catch(Exception e) {			
            PrintUtilities.abort(e);
		}
				
		System.exit(0); //TODO this is a hot fix for not properly shutting down executor service in ParallelFor class
	}

	private static void parseCMD(String[] args) throws Exception {
		//some default settings are specified in the AppContext class but here we
		//override the default settings using the provided command line arguments		
		for (int i = 0; i < args.length; i++) {
			String[] options = args[i].trim().split("=");
			
			if ((! options[0].equals("-ucr")) && (! options[0].equals("-train")) && (! options[0].equals("-test"))  && (! options[0].equals("-out"))) {
				options[1] = options[1].trim().toLowerCase();
			}
			
			switch(options[0]) {

			case "-v":
			case "-version":
				System.out.println("Version: " + AppContext.version);
				System.out.println("VersionTag: " + AppContext.version_tag);
				System.out.println("Release Notes: " + AppContext.release_notes);
				
				//TODO print java version info and slurm environment info here
				
				System.exit(0);
			case "-h":
			case "-help":
				System.out.println("Version: " + AppContext.version);
				//TODO
				System.out.println("TODO print help message -- and example of command line args: " + AppContext.version);

				System.exit(0);
			case "-train":
				AppContext.training_file = options[1];
				break;
			case "-test":
				AppContext.testing_file = options[1];
				break;
			case "-validate":
				AppContext.validate_file = options[1];
				break;
			case "-feature_file":
				AppContext.feature_save_file = options[1];
				break;
			case "-out":
				AppContext.output_dir = options[1];
				break;
			case "-repeats":
				AppContext.num_repeats = Integer.parseInt(options[1]);
				break;
			case "-trees":
				AppContext.num_trees = Integer.parseInt(options[1]);
				break;
			case "-s": //use either -s or -s_prob, dont use both params together
				//NOTE eg. s=ee:5,boss:100
				AppContext.use_probabilities_to_choose_num_splitters_per_node = false;
				parseNumSplittersPerNode(options[1], false);
				break;
			case "-threads":
				AppContext.num_threads = Integer.parseInt(options[1]);
				break;
			case "-export":
				AppContext.export_level =  Integer.parseInt(options[1]);
				break;
			case "-verbosity":
				AppContext.verbosity =  Integer.parseInt(options[1]);
				break;

			case "-randf_m":
				
				if (options[1].equals("sqrt")) {
					AppContext.randf_feature_selection = FeatureSelectionMethod.Sqrt;
				}else if (options[1].equals("loge")) {
					AppContext.randf_feature_selection = FeatureSelectionMethod.Loge;
				}else if (options[1].equals("log2")) {
					AppContext.randf_feature_selection = FeatureSelectionMethod.Log2; 
				}else {
					AppContext.randf_feature_selection = FeatureSelectionMethod.ConstantInt; 
					AppContext.randf_m = Integer.parseInt(options[1]); //TODO verify
				}	
				
				break;
			case "-boss_params":
				if (options[1].equals("random")) {
					AppContext.boss_param_selection = ParamSelection.Random;
				}else if (options[1].equals("best")) {
					AppContext.boss_param_selection = ParamSelection.PreLoadedSet;
				}else {
					throw new Exception("Boss param selection method not suppoted");
				}
				break;
			case "-boss_trasformations":
				AppContext.boss_trasformations = Integer.parseInt(options[1]);
				break;
			case "-boss_preload_params_path":
				AppContext.boss_params_files = options[1];
				break;
			case "-boss_transform_level":
				if (options[1].equals("forest")) {
					AppContext.boss_transform_level = TransformLevel.Forest;
				}else if (options[1].equals("tree")) {
					AppContext.boss_transform_level = TransformLevel.Forest;
				}else {
					throw new Exception("Boss transform level not suppoted");
				}
				break;
			case "-boss_split_method":
				if (options[1].equals("gini")) {
					AppContext.boss_split_method = SplitMethod.Binary_Gini;
				}else if (options[1].equals("nearest")) {
					AppContext.boss_split_method = SplitMethod.Nary_NearestClass;
				}else {
					throw new Exception("Boss split method not suppoted");
				}
				break;
			case "-rif_min_interval":
				AppContext.rif_min_interval = Integer.parseInt(options[1]); //TODO validate
				break;
			case "-rif_components":
				if (options[1].equals("acf")) {
					AppContext.rif_components = RifFilters.ACF;
				}else if (options[1].equals("pacf")) {
					AppContext.rif_components = RifFilters.PACF;
				}else if (options[1].equals("arma")) {
					AppContext.rif_components = RifFilters.ARMA;
				}else if (options[1].equals("ps")) {
					AppContext.rif_components = RifFilters.PS;
				}else if (options[1].equals("dft")) {
					AppContext.rif_components = RifFilters.DFT;
				}else if (options[1].equals("acf_pacf_arma")) {
					AppContext.rif_components = RifFilters.ACF_PACF_ARMA;
				}else if (options[1].equals("acf_pacf_arma_ps_comb")) {
					AppContext.rif_components = RifFilters.ACF_PACF_ARMA_PS_combined;
				}else if (options[1].equals("acf_pacf_arma_ps_sep")) {
					AppContext.rif_components = RifFilters.ACF_PACF_ARMA_PS_separately;
				}else if (options[1].equals("acf_pacf_arma_dft")) {
					AppContext.rif_components = RifFilters.ACF_PACF_ARMA_DFT;
				}else {
					throw new Exception("RISE component not suppoted");
				}
				break;
			case "-rif_m":
				if (options[1].equals("sqrt")) {
					AppContext.rif_feature_selection = FeatureSelectionMethod.Sqrt;
				}else if (options[1].equals("loge")) {
					AppContext.rif_feature_selection = FeatureSelectionMethod.Loge;
				}else if (options[1].equals("log2")) {
					AppContext.rif_feature_selection = FeatureSelectionMethod.Log2; 
				}else {
					AppContext.rif_feature_selection = FeatureSelectionMethod.ConstantInt; 
					AppContext.rif_m = Integer.parseInt(options[1]); //TODO verify
				}	
				break;
			case "-rif_same_intv_component":
				AppContext.rif_same_intv_component =  Boolean.parseBoolean(options[1]);
				break;
			case "-rif_num_intervals":
				AppContext.rif_num_intervals = Integer.parseInt(options[1]); //TODO validate
				break;
			case "-binary_split":
				AppContext.binary_split =  Boolean.parseBoolean(options[1]);
				break;
			case "-gini_split":
				AppContext.gini_split =  Boolean.parseBoolean(options[1]);
				break;
			default:
				throw new Exception("Invalid Commandline Argument: " + args[i]);
			}
		}
		
		//NOTE do these after all params have been parsed from cmd line
		
		//extra validations
		if (AppContext.boss_param_selection == ParamSelection.PreLoadedSet && ! use_ucr) {
			throw new Exception("Cross validated BOSS params are only available for UCR datasets");
		}
		
		if(AppContext.tsf_enabled) {
			//TODO tested only for 1 interval per splitter. check for more intervals
			AppContext.num_actual_tsf_splitters_needed = AppContext.tsf_splitters_per_node / 3;

		}
		
		if(AppContext.rif_enabled) {
			//TODO tested only for 1 interval per splitter. check for more intervals

			int num_gini_per_type = AppContext.rif_splitters_per_node / 4;	// eg. if we need 12 gini per type 12 ~= 50/4 
			int extra_gini = AppContext.rif_splitters_per_node % 4;
			//assume 1 interval per splitter
			// 2 = ceil(12 / 9) if 9 = min interval length
			int min_splitters_needed_per_type = (int) Math.ceil((float)num_gini_per_type / (float)AppContext.rif_min_interval); 
			int max_attribs_to_use_per_splitter = (int) Math.ceil(num_gini_per_type / min_splitters_needed_per_type);

			AppContext.num_actual_rif_splitters_needed_per_type = min_splitters_needed_per_type;
			AppContext.rif_m = max_attribs_to_use_per_splitter;
			int approx_gini_estimated = 4 * max_attribs_to_use_per_splitter * min_splitters_needed_per_type;
//			System.out.println("RISE: approx_gini_estimated: " + approx_gini_estimated); 
			
		}
		
		if (!(AppContext.rif_components == RifFilters.ACF_PACF_ARMA_PS_combined)) {
			AppContext.num_actual_splitters_needed = 
					AppContext.ee_splitters_per_node +
					AppContext.randf_splitters_per_node +
					AppContext.rotf_splitters_per_node +
					AppContext.st_splitters_per_node +
					AppContext.boss_splitters_per_node +
					AppContext.sax_splitters_per_node +
					AppContext.num_actual_tsf_splitters_needed +
					AppContext.num_actual_rif_splitters_needed_per_type;	//TODO works if 
		}else {
			AppContext.num_actual_splitters_needed = 
					AppContext.ee_splitters_per_node +
					AppContext.randf_splitters_per_node +
					AppContext.rotf_splitters_per_node +
					AppContext.st_splitters_per_node +
					AppContext.boss_splitters_per_node +
					AppContext.sax_splitters_per_node +
					AppContext.num_actual_tsf_splitters_needed +
					AppContext.rif_splitters_per_node;	//TODO works if
		}
		

				
		if(AppContext.num_splitters_per_node == 0) {
			throw new Exception("Number of candidate splits per node must be greater than 0. use -s option");
		}
		
	}
	
	
	
	//eg. -s=10  , -s=ee:5,boss:100   , -s_prob=10,ee:0.1   ,-s_prob=10,equal
	private static void parseNumSplittersPerNode(String string, boolean use_probabilities) throws Exception {
		ArrayList<SplitterType> splitters = new ArrayList<>();
		String[] options = string.split(",");
		
		if (use_probabilities) {	//TODO does not work -experimental
			//TODO exception handling
			
			//assume first item is an integer 
			AppContext.num_splitters_per_node = Integer.parseInt(options[0]);
			
			//parse probabilities
			double total = 0;
			
			for (int i = 1; i < options.length; i++) {	//TODO handle boundary conditions
				String temp[] = options[i].split(":");
				
				//TODO if equal ...				
				if (temp[0].equals("ee")) {
					AppContext.probability_to_choose_ee = Double.parseDouble(temp[1]);
					total += AppContext.probability_to_choose_ee;
					if (AppContext.probability_to_choose_ee > 0) {
						AppContext.ee_enabled = true;
						splitters.add(SplitterType.ElasticDistanceSplitter);
					}
				}else if (temp[0].equals("randf")) {
					AppContext.probability_to_choose_randf = Double.parseDouble(temp[1]);
					total += AppContext.probability_to_choose_randf;
					if (AppContext.probability_to_choose_randf > 0) {
						AppContext.randf_enabled = true;
						splitters.add(SplitterType.RandomForestSplitter);
					}
				}else if (temp[0].equals("rotf")) {
					AppContext.probability_to_choose_rotf = Double.parseDouble(temp[1]);
					total += AppContext.probability_to_choose_rotf;
					if (AppContext.probability_to_choose_rotf > 0) {
						AppContext.rotf_enabled = true;
						splitters.add(SplitterType.RotationForestSplitter);
					}
				}else if (temp[0].equals("st")) {
					AppContext.probability_to_choose_st = Double.parseDouble(temp[1]);
					total += AppContext.probability_to_choose_st;
					if (AppContext.probability_to_choose_st > 0) {
						AppContext.st_enabled = true;
						splitters.add(SplitterType.ShapeletTransformSplitter);
					}
				}else if (temp[0].equals("boss")) {
					AppContext.probability_to_choose_boss = Double.parseDouble(temp[1]);
					total += AppContext.probability_to_choose_boss;
					if (AppContext.probability_to_choose_boss > 0) {
						AppContext.boss_enabled = true;
						splitters.add(SplitterType.BossSplitter);
					}
				}else if (temp[0].equals("tsf")) {
					AppContext.probability_to_choose_tsf = Double.parseDouble(temp[1]);
					total += AppContext.probability_to_choose_tsf;
					if (AppContext.probability_to_choose_tsf > 0) {
						AppContext.tsf_enabled = true;
						splitters.add(SplitterType.TSFSplitter);
					}
				}else if (temp[0].equals("rif")) {
					AppContext.probability_to_choose_rif = Double.parseDouble(temp[1]);
					total += AppContext.probability_to_choose_rif;
					if (AppContext.probability_to_choose_rif > 0) {
						AppContext.rif_enabled = true;
						splitters.add(SplitterType.RIFSplitter);
					}
				}else {
					throw new Exception("Unknown Splitter Type");
				}
				
			}
			
			//override the last one
			if (total > 1) {
				throw new Exception("Probabilities add up to more than 14");
			}
			
		}else {
			
			int total = 0;
			
			for (int i = 0; i < options.length; i++) {	//TODO handle boundary conditions
				String temp[] = options[i].split(":");
				
				if (temp[0].equals("ee")) {
					AppContext.ee_splitters_per_node = Integer.parseInt(temp[1]);
					total += AppContext.ee_splitters_per_node;
					if (AppContext.ee_splitters_per_node > 0) {
						AppContext.ee_enabled = true;
						splitters.add(SplitterType.ElasticDistanceSplitter);
					}
				}else if (temp[0].equals("randf")) {
					AppContext.randf_splitters_per_node = Integer.parseInt(temp[1]);
					total += AppContext.randf_splitters_per_node;
					if (AppContext.randf_splitters_per_node > 0) {
						AppContext.randf_enabled = true;
						splitters.add(SplitterType.RandomForestSplitter);
					}
				}else if (temp[0].equals("rotf")) {
					AppContext.rotf_splitters_per_node = Integer.parseInt(temp[1]);
					total += AppContext.probability_to_choose_rotf;
					if (AppContext.probability_to_choose_rotf > 0) {
						AppContext.rotf_enabled = true;
						splitters.add(SplitterType.RotationForestSplitter);
					}
				}else if (temp[0].equals("st")) {
					AppContext.st_splitters_per_node = Integer.parseInt(temp[1]);
					total += AppContext.st_splitters_per_node;
					if (AppContext.st_splitters_per_node > 0) {
						AppContext.st_enabled = true;
						splitters.add(SplitterType.ShapeletTransformSplitter);
					}
				}else if (temp[0].equals("boss")) {
					AppContext.boss_splitters_per_node = Integer.parseInt(temp[1]);
					total += AppContext.boss_splitters_per_node;
					if (AppContext.boss_splitters_per_node > 0) {
						AppContext.boss_enabled = true;
						splitters.add(SplitterType.BossSplitter);
					}
				}else if (temp[0].equals("tsf")) {
					AppContext.tsf_splitters_per_node = Integer.parseInt(temp[1]);
					total += AppContext.tsf_splitters_per_node;
					if (AppContext.tsf_splitters_per_node > 0) {
						AppContext.tsf_enabled = true;
						splitters.add(SplitterType.TSFSplitter);
					}
				}else if (temp[0].equals("rif") || temp[0].equals("rise")) {
					AppContext.rif_splitters_per_node = Integer.parseInt(temp[1]);				
					total += AppContext.rif_splitters_per_node;
					if (AppContext.rif_splitters_per_node > 0) {
						AppContext.rif_enabled = true;
						splitters.add(SplitterType.RIFSplitter);
					}
				}else if (temp[0].equals("sax")) {
					AppContext.sax_splitters_per_node = Integer.parseInt(temp[1]);
					total += AppContext.sax_splitters_per_node;
					if (AppContext.sax_splitters_per_node > 0) {
						AppContext.sax_enabled = true;
						splitters.add(SplitterType.SAXSplitter);
					}
				}else {
					throw new Exception("Unknown Splitter Type");
				}
			}
			
			AppContext.num_splitters_per_node = total;
			AppContext.enabled_splitters = splitters.toArray(new SplitterType[splitters.size()]);
			
		}
	}

	
}

