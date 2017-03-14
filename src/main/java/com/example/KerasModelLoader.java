package com.example;


import java.io.File;
import java.io.IOException;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class KerasModelLoader {
	private static final Logger log = LoggerFactory.getLogger(KerasModelLoader.class);
	
	public static void main(String[] args) throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException, InterruptedException {
		int skipNumLines = 0;
		String delimiter = ",";
		
		MultiLayerNetwork network = KerasModelImport.importKerasSequentialModelAndWeights("/tmp/iris_model.json", "/tmp/iris_weights.h5");
		CSVRecordReader reader = new CSVRecordReader(skipNumLines, delimiter);
		reader.initialize(new FileSplit(new File("src/main/resources/records.csv")));
		DataSetIterator iter = new RecordReaderDataSetIterator(reader, 150, 4, 3);

		DataSet set = iter.next();
		set.shuffle();
		INDArray output = network.output(set.getFeatureMatrix());
		
		Evaluation eval = new Evaluation();
		eval.eval(set.getLabels(), output);
		log.info(eval.stats());
	}
}
