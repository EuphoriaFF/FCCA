package feature;

public class FeatureRISE extends Feature {

    public int attributeValue;
    public int intervalSt;
    public int intervalEd;
    public String tramsformType;

    public FeatureRISE(int featureId, String featureType, int representLabel, double infogain, double threshold, int sign, int attributeValue, int intervalSt, int intervalEd, String tramsformType) {
        super(featureId, featureType, representLabel, infogain, threshold, sign);
        this.attributeValue = attributeValue;
        this.intervalSt = intervalSt;
        this.intervalEd = intervalEd;
        this.tramsformType = tramsformType;
    }

    @Override
    public String toString() {
        StringBuilder ss = new StringBuilder();
        ss.append(featureId + "," + featureType + "," + representLabel + "," + infogain + "," + threshold + "," + sign);
        ss.append(","+attributeValue + ","+intervalSt+","+intervalEd+","+tramsformType);
        ss.append("\n");

        return ss.toString();
    }

}
