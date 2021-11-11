package feature;

import datasets.BossDataset;

public class FeatureSFA extends Feature {

    public int SFAvalue; // SFA值
    public String SFA_abc; //原始SFA表示
    public int origin_ts_len; //boss参数中的window_len

    public int origin_ts_Id;
    public BossDataset.BossParams boss_params; //当前sfa来源boss的参数信息

    public int origin_ts_st;

    public FeatureSFA(int featureId, String featureType, int representLabel, double infogain, double threshold, int sign, int SFAvalue, String SFA_abc, int origin_ts_len, int origin_ts_Id, BossDataset.BossParams boss_params, int origin_ts_st) {
        super(featureId, featureType, representLabel, infogain, threshold, sign);
        this.SFAvalue = SFAvalue;
        this.SFA_abc = SFA_abc;
        this.origin_ts_len = origin_ts_len;
        this.origin_ts_Id = origin_ts_Id;
        this.boss_params = boss_params;
        this.origin_ts_st = origin_ts_st;
    }

    @Override
    public String toString() {

        String[] split = this.SFA_abc.substring(1,this.SFA_abc.length()-1).split(", ");
        String sfaword_abc = "";
        for (String s : split) {
            sfaword_abc+= (char)('a'+Integer.parseInt(s));
        }

        StringBuilder ss = new StringBuilder();
        ss.append(featureId + "," + featureType + "," + representLabel + "," + infogain + "," + threshold+ "," + sign);
        ss.append(","+SFAvalue+","+sfaword_abc+","+boss_params.window_len);
        ss.append(","+ origin_ts_Id + ","+boss_params.normMean+","+boss_params.word_len+","+boss_params.alphabet_size);
        ss.append(","+origin_ts_st);
        ss.append("\n");

        return ss.toString();
    }
}
