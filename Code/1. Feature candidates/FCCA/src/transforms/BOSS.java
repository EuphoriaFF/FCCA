// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)

// modified by shifaz 2018
package transforms;

import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicInteger;

import com.carrotsearch.hppc.IntIntHashMap;
import com.carrotsearch.hppc.cursors.IntIntCursor;

import core.ParallelFor;
import datasets.TimeSeries;
import transforms.SFA.HistogramType;
import transforms.SFA.Words;

/**
 * The Bag-of-SFA-Symbols boss as published in
 * Schäfer, P.: The boss is concerned with time series classification
 * in the presence of noise. DMKD 29(6) (2015) 1505–1530
 *
 * @author bzcschae
 */
public class BOSS {

  public int symbols;
  public int maxF;
  public int windowLength;
  public boolean normMean;
  public SFA signature;

  public final static int BLOCKS;

  static {
    Runtime runtime = Runtime.getRuntime();
    if (runtime.availableProcessors() <= 4) {
      BLOCKS = 8;
    } else {
      BLOCKS = runtime.availableProcessors();
    }

    //    BLOCKS = 1; // for testing purposes
  }

  public BOSS(){}

  /**
   * Create a BOSS boss.
   *
   * @param maxF         queryLength of the SFA words
   * @param maxS         alphabet size
   * @param windowLength sub-sequence (window) queryLength used for extracting SFA words from
   *                     time series.
   * @param normMean     set to true, if mean should be set to 0 for a window
   */
  public BOSS(int maxF, int maxS, int windowLength, boolean normMean) {
    this.maxF = maxF;
    this.symbols = maxS;
    this.windowLength = windowLength;
    this.normMean = normMean;
  }

  /**
   * The BOSS boss: a histogram of SFA word frequencies
   */
	public static class BagOfPattern {
		public IntIntHashMap bag;

		private Double label;
		// TODO add a pointer to the time series ?? too much memory?? think, might
		// simplify somethings
		private TimeSeries series;

		public ArrayList<String> sfa_ls;

      public int getSeriesId() {
          return seriesId;
      }

      private int seriesId;

		public BagOfPattern() {
			this.bag = new IntIntHashMap();
		}

//    public BagOfPattern(int size, Integer label) {
//      this.bag = new IntIntHashMap(size);
//      this.label = label.dblValue();
//    }

		public BagOfPattern(int size, TimeSeries series, int tsId, ArrayList<String> sfa_ls) {
          this.bag = new IntIntHashMap(size);
          this.label = series.getLabel().doubleValue();
          this.series = series;
          this.seriesId = tsId;
          this.sfa_ls = sfa_ls;
      }

		public IntIntHashMap getBag() {
			return bag;
		}

		public Integer getLabel() {
			return label.intValue();
		}
		
		public void setLabel(Integer label) {
			this.label = label.doubleValue(); 	//TODO FIXME check if it matches series label if this is a series hist. 
		}

		public TimeSeries getSeries() {
			return series;
		}

		public String toString() {

		    return this.label + "=" + this.bag.toString();
		}

//    public String toString() {
//	    final long mask = (1L << (usedBits * wordLength)) - 1L;
//
//		StringBuilder sfaWord = new StringBuilder();
//
//		for (IntIntCursor key : this.bag) {
//			words[j] = (int) Words.createWord(sfaWords[j], windows[i], (byte) Words.binlog(maxS));
//
//			sfaWord.append((char) (Character.valueOf('a') + key.key));
//		}
//
//    	return sfaWord.toString();
//    }
//
//
//	public static String toSfaWord(short[] word, int symbols) {
//		StringBuilder sfaWord = new StringBuilder();
//
//		for (short c : word) {
//			sfaWord.append((char) (Character.valueOf('a') + c));
//		}
//
//		return sfaWord.toString();
//	}
  }

  /**
   * Create SFA words for all samples
   *
   * @param samples the time series to be transformed
   * @return returns an array of words for each time series
   */
  public int[][] createWords(final TimeSeries[] samples) {

    final int[][] words = new int[samples.length][];

    if (this.signature == null) {
      this.signature = new SFA(HistogramType.EQUI_DEPTH);
      this.signature.fitWindowing(samples, this.windowLength, this.maxF, this.symbols, this.normMean, true);
    }

    // create sliding windows
    ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {
      @Override
      public void run(int id, AtomicInteger processed) {
        for (int i = 0; i < samples.length; i++) {
          if (i % BLOCKS == id) {
            short[][] sfaWords = BOSS.this.signature.transformWindowing(samples[i]);
            words[i] = new int[sfaWords.length];
            for (int j = 0; j < sfaWords.length; j++) {
              words[i][j] = (int) Words.createWord(sfaWords[j], BOSS.this.maxF, (byte) Words.binlog(BOSS.this.symbols));
            }
          }
        }
      }
    });

    return words;
  }
}
