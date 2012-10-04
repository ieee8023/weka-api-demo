package joecohen.weka;

import java.io.File;
import java.io.FileWriter;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

import com.csvreader.CsvReader;

/**
 * 
 * @author Joseph Paul Cohen
 * @category Examples, Weka\
 * @version 1
 * @date 12/7/2011
 *
 *	This code is designed as a code sample to demo how to create an Instances object
 *	and then use it to classify via API or export to an arff file.
 */

public class CsvToArff {
	
	/**
	 * This file will convert a csv file into the arff format. 
	 * It will then perform 10 fold cross validation to find the F1
	 * scores for three classifiers applied to this dataset.
	 * 
	 * @param args
	 * @throws Exception
	 */
	
	public static void main(String[] args) throws Exception {
		
		/*
		 * We first declare what our input and out files are going to be
		 * We assume that the csv file has headers, all numerical values,
		 * and the class label at the last column
		 */
		String inputFileName = "testAttributeSet.csv";
		String outputFileName = "testAttributeSet.arff";
		boolean hasHeaders = true;
		
		/*
		 * We find two important values here.
		 * The record count is the numnber is instances we will create.
		 * The classIndex is the column that contains class information.
		 */
		int recordCount = 0;
		int classIndex = 0;
		CsvReader reader = new CsvReader(inputFileName);
		if (hasHeaders) reader.readHeaders();
		while (reader.readRecord()){
			classIndex = reader.getColumnCount()-1;
			recordCount++;
		}
		
		/*
		 * Here we enumerate the classes at the class index location into
		 * the set classes.
		 */
		Set<String> classes = new HashSet<String>();
		reader = new CsvReader(inputFileName);
		if (hasHeaders) reader.readHeaders();
		while (reader.readRecord()){
			String val = reader.get(classIndex);
			classes.add(val);
		}
		
		/*
		 * The classes we just extracted need to be converted into a format Weka
		 * can understand.  We create a FastVector to size of the number of classes
		 * plus one to account for the unknown class 0.
		 */
		FastVector fvClass = new FastVector(classes.size()+1);
		fvClass.addElement("0"); // add unknown class
		for(String s : classes)
			fvClass.addElement(s);
		
		// Show which classes we found
		System.out.println("Classes: ");
		for (Object s: fvClass.toArray())
			System.out.println("  " + s);
		
		/*
		 * We now create an attribute vector that Weka will use to identify
		 * each column's type. We have to add these in order.
		 */
		FastVector fvWekaAttributes = new FastVector(classIndex);
		reader = new CsvReader(inputFileName);
		if (hasHeaders) reader.readHeaders();
		reader.readRecord();
		for (int i = 0; i < reader.getColumnCount() ; i++){
			String name = reader.getHeader(i);
			if (name.equals("")) name = "attr" + i;
			if (i == classIndex){
				fvWekaAttributes.addElement(new Attribute(name,fvClass));
			}else{
				String val = reader.get(i);
				try{
					Double.parseDouble(val);
					fvWekaAttributes.addElement(new Attribute(name));
				}catch(Exception e){
					throw new Exception("String fields are not supported in this code, maybe hasHeaders should be marked: " + val);
	
				}
			}
		}
		System.out.println("Attributes: ");
		for (Object s: fvWekaAttributes.toArray())
			System.out.println("  " + s);
		
		/*
		 * We now populate the instances by filling it with instance objects. We create 
		 * a new instances using the attributes we just populated and then set the class index.
		 * We declare a new instance of size classindex+1 to make room for the class index.
		 * We then add each column to the instance using it's attribute object we added before.
		 */
		Instances instances = new Instances(inputFileName, fvWekaAttributes, recordCount); 
		instances.setClassIndex(classIndex);
		reader = new CsvReader(inputFileName);
		if (hasHeaders) reader.readHeaders();
		while (reader.readRecord()){
			Instance instance = new Instance(classIndex+1);
			for (int i = 0; i < reader.getColumnCount(); i++){
				String val = reader.get(i);
				try{
					Double ival = Double.parseDouble(val);
					instance.setValue((Attribute)fvWekaAttributes.elementAt(i), ival);
				}catch(Exception e2){
					throw new Exception("String fields are not supported in this code: Column " + i);
				}
			}
			instances.add(instance);
		}
		
		
		/*
		 * We now write our arff file to a file by calling the toString method 
		 * on the instances object.
		 */

		FileWriter output = new FileWriter(new File(outputFileName));
		output.write(instances.toString());
		output.flush();
		output.close();

		 
		 
		/*
		 * We now compute F1 using different classifiers.
		 */
		
		performCrossValidation(new J48(), instances);
		performCrossValidation(new NaiveBayes(), instances);
		performCrossValidation(new LibSVM(), instances);
		 
	}

	static void performCrossValidation(Classifier classifier, Instances instances) throws Exception{
		
		classifier.buildClassifier(instances);
		Evaluation eval = new Evaluation(instances);
		eval.crossValidateModel(classifier, instances, 10, new Random());
		System.out.println("Using classifier: " + classifier.getClass().getCanonicalName());
		System.out.println("  Number of instances in set: " + eval.numInstances());
		System.out.println("  Kappa Value is: " + eval.kappa());
		System.out.println("  Error Rate is: " + eval.errorRate());
		System.out.println("  F1 Score is: " + eval.weightedFMeasure());
		
	}
	
}
