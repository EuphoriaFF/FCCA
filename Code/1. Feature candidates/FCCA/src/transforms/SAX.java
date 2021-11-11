package transforms;

import datasets.TimeSeries;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.HashMap;

public class SAX {

    //每个时间序列数据对应的一个直方图
    public static class BagOfPattern{
        private HashMap<String, Double> bag;
        private int label;
        private TimeSeries series;
        private int seriesId;

        public ArrayList<String> sax_ls;

        public BagOfPattern(HashMap<String, Double> bag, int label, TimeSeries series, int seriesId, ArrayList<String> sax_ls) {
            this.bag = bag;
            this.label = label;
            this.series = series;
            this.seriesId = seriesId;
            this.sax_ls = sax_ls;
        }

        public int getLabel() {
            return label;
        }

        public TimeSeries getSeries() {
            return series;
        }

        public int getSeriesId() {
            return seriesId;
        }

        @Override
        public String toString() {
            return "BagOfPattern{" +
                    "bag=" + bag +
                    ", label=" + label +
                    ", series=" + series +
                    ", seriesId=" + seriesId +
                    '}';
        }
    }

}
