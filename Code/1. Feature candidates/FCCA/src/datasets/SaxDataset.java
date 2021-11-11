package datasets;

import transforms.SAX;
import transforms.SAX.BagOfPattern;

import java.util.HashMap;

public class SaxDataset {

    public BagOfPattern[] sax_transformed_data;
    public SAXParams saxParams;

    public SaxDataset(BagOfPattern[] sax_transformed_data, SAXParams saxParams) {
        this.sax_transformed_data = sax_transformed_data;
        this.saxParams = saxParams;
    }

    public static class SAXParams{
        public int alphabet_size = 4; //字母表大小
        public int window_len = 10; //窗口长度
        public int Paa_size = 10; //PAA分段个数
        public double nThreshold = 0.001; //进行标准化的阈值

        public SAXParams(int alphabet_size, int window_len, int paa_size) {
            this.alphabet_size = alphabet_size;
            this.window_len = window_len;
            Paa_size = paa_size;
            this.nThreshold = 0.001;
        }

        public SAXParams(int alphabet_size, int window_len, int paa_size, double nThreshold) {
            this.alphabet_size = alphabet_size;
            this.window_len = window_len;
            Paa_size = paa_size;
            this.nThreshold = nThreshold;
        }

        @Override
        public String toString() {
            return "SAXParams{" +
                    "alphabet_size=" + alphabet_size +
                    ", window_len=" + window_len +
                    ", Paa_size=" + Paa_size +
                    ", nThreshold=" + nThreshold +
                    '}';
        }
    }

    public void print_sax_transformed_data(){
        for (BagOfPattern bag : sax_transformed_data) {
            System.out.println(bag);
        }
    }

}
