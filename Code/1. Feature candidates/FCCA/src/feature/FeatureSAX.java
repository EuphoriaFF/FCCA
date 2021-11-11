package feature;

import datasets.SaxDataset;

public class FeatureSAX extends Feature {

    public String sax_word;
    public int origin_ts_Id;
    public SaxDataset.SAXParams sax_params;
    public int origin_ts_st;

    public FeatureSAX(int featureId, String featureType, int representLabel, double infogain, double threshold, int sign, String sax_word, int origin_ts_Id , SaxDataset.SAXParams sax_params, int origin_ts_st) {
        super(featureId, featureType, representLabel, infogain, threshold, sign);
        this.sax_word = sax_word;
        this.origin_ts_Id = origin_ts_Id;
        this.origin_ts_st = origin_ts_st;
        this.sax_params = sax_params;
    }

    public FeatureSAX(int featureId, String featureType, int representLabel, double infogain, double threshold, int sign, String sax_word, int origin_ts_Id , SaxDataset.SAXParams sax_params) {
        super(featureId, featureType, representLabel, infogain, threshold, sign);
        this.sax_word = sax_word;
        this.origin_ts_Id = origin_ts_Id;
        this.origin_ts_st = -1;
        this.sax_params = sax_params;
    }

    @Override
    public String toString() {
        StringBuilder ss = new StringBuilder();
        ss.append(featureId + "," + featureType + "," + representLabel + "," + infogain + "," + threshold+ "," + sign);
        ss.append(","+sax_word);
        ss.append(","+ origin_ts_Id + ","+sax_params.window_len+","+sax_params.Paa_size+","+sax_params.alphabet_size);
        ss.append(","+origin_ts_st);
        ss.append("\n");

        return ss.toString();
    }
}
