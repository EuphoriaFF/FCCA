package feature;

public class Feature {
    public int featureId;
    public String featureType;
    public int representLabel;
    public double infogain;
    public double threshold;
    public int sign;

    public Feature(int featureId, String featureType, int representLabel, double infogain, double threshold, int sign) {
        this.featureId = featureId;
        this.featureType = featureType;
        this.representLabel = representLabel;
        this.infogain = infogain;
        this.threshold = threshold;
        this.sign = sign;
    }

    @Override
    public String toString() {
        return featureId + "," + featureType + "," + representLabel + "," + infogain + "," + threshold + "," + sign +"\n";
    }
}
