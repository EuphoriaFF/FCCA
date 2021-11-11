package datasets;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.TreeMap;

import com.carrotsearch.hppc.cursors.IntCursor;
import com.carrotsearch.hppc.cursors.IntIntCursor;

import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import transforms.BOSS.BagOfPattern;

public class BossDataset {

	private BagOfPattern[] transformed_data;
	
	private BossParams params;
	
	//TODO
//	make a histogram for the whole dataset
	
	public static class BossParams {
		public boolean normMean;
		public int window_len;
		public int word_len;
		public int alphabet_size;
		public boolean lower_bounding = true;	//TODO
		
		public BossParams(boolean norm, int w, int l, int a) {
			this.normMean = norm;
			this.window_len = w;
			this.word_len = l;
			this.alphabet_size = a;
		}
		
		public String toString() {
			return "" + "," + normMean + ","+ window_len + ","+ word_len + ","+ alphabet_size;
		}

		public String toSFAString() {
	return normMean + ","+ window_len + ","+ word_len + ","+ alphabet_size;
}
	}

	
	public BossDataset(TSDataset original_dataset,BagOfPattern[] transformed_data, BossParams params) {
//		this.original_dataset = original_dataset;
		this.transformed_data = transformed_data;
		this.params = params;
	}
	
	public TIntObjectMap<List<BagOfPattern>> split_by_class() {
		TIntObjectMap<List<BagOfPattern>> split =  new TIntObjectHashMap<List<BagOfPattern>>();
		Integer label;
		List<BagOfPattern> class_set = null;

		for (int i = 0; i < this.transformed_data.length; i++) {
			label = this.transformed_data[i].getLabel();
			if (! split.containsKey(label)) {
				class_set = new ArrayList<BagOfPattern>();
				split.put(label, class_set);
			}
			
			split.get(label).add(this.transformed_data[i]);
		}
		
		return split;
	}
	
	
	public BagOfPattern[] getTransformed_data() {
		return transformed_data;
	}


	public String toString() {
		StringBuilder sb = new StringBuilder();
		
		sb.append("all: ");
		sb.append(Arrays.toString( this.transformed_data));
		sb.append("\n");
		
		TIntObjectMap<List<BagOfPattern>> split = split_by_class();
		
		for (int key : split.keys()) {
			sb.append("class " + key +  ":");
			
			BagOfPattern[] array = split.get(key).toArray(new BagOfPattern[split.get(key).size()]);
			
			sb.append(Arrays.toString(array));
			sb.append("\n");
		}
				
		return sb.toString();
	}

}
