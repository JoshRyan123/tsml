package Laboratories.lab4;

import evaluation.tuning.ParameterSpace;
import experiments.Experiments;
import experiments.data.DatasetLoading;
import fileIO.OutFile;
import machine_learning.classifiers.tuned.TunedClassifier;
import utilities.InstanceTools;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.*;
import weka.classifiers.trees.*;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Random;

import static org.apache.log4j.NDC.clear;

public class Whiskylab {
    public static String basePath="C:\\Work\\GitHub\\tsml\\Data\\Labs\\";

    static String[] allClassifiers={"NaiveBayes","BayesNet", "AODE", "AODEsr", "BayesianLogisticRegression", "ComplementNaiveBayes", "DMNBtext", "NaiveBayesSimple", "NaiveBayesUpdateable"};
    static String[] allProblems ={"WhiskyData"};

    public static Classifier setClassifier(String c){
        switch(c){
            case "NaiveBayes":
                return new NaiveBayes();
            case "BayesNet":
                return new BayesNet();
            case "AODE":
                return new AODE();
            case "AODEsr":
                return new AODEsr();
            case "BayesianLogisticRegression":
                return new BayesianLogisticRegression();
            case "ComplementNaiveBayes":
                return new ComplementNaiveBayes();
            case "DMNBtext":
                return new DMNBtext();
            case "NaiveBayesSimple":
                return new NaiveBayesSimple();
            case "NaiveBayesUpdateable":
                return new NaiveBayesUpdateable();
        }
        return null;
    }

    public static void runExperimentManually() throws Exception {

        String problem = "WhiskyData";
        Instances data = DatasetLoading.loadData(basePath+problem+"/"+problem+".arff");

        System.out.println(data.numClasses());

        Instances[] split = InstanceTools.resampleInstances(data, 0, 0.5);
        //
        //split = DatasetLoading.sampleHayesRoth(0);

        Classifier c = new NaiveBayes();
        c.buildClassifier(split[0]);


        OutFile out = new OutFile("C:/Work/GitHub/tsml/Results/"+problem+"Resample0.csv");
        out.writeLine(c.getClass().getSimpleName()+","+problem);

        out.writeLine("No parameter info");
        out.writeLine("Blank");

        for(Instance ins:split[1]){
            //Inefficient to call twice
            int pred = (int)c.classifyInstance(ins);
            double[] probs = c.distributionForInstance(ins);

            // print class value predictions
            out.writeString((int)ins.classValue()+","+pred+",");
            System.out.print((int)ins.classValue()+","+pred+",");

            // print probability distrubutions
            for(double d:probs) {
                System.out.print("," + d);
                out.writeString("," + d);
            }

            System.out.print("\n");
            out.writeString("\n");
        }

        // Evaluate
        Evaluation eval = new Evaluation(split[0]);

        eval.evaluateModel(c,split[1]);

        // finding accuracy
        double acc =1-eval.errorRate();

        // area under the ROC curve
        double weightedAuroc = eval.weightedAreaUnderROC();

        System.out.println(" Acc = "+acc+" auroc = "+weightedAuroc);

        // inspect this method in weka.classifiers.evaluation
        eval.crossValidateModel(c,data,10,new Random());

        //weightedAuroc = eval.areaUnderROC(split[0].classIndex());

        System.out.println(" Acc = "+acc+" auroc = "+weightedAuroc);
    }


    public static void runExperimentAutomatically() throws Exception {
    }


    public static void evaluateInCode() throws Exception {
    }



    public static void runMultipleExperiments() throws Exception {
        // create expSettings instance for parameter setting
        Experiments.ExperimentalArguments expSettings = new Experiments.ExperimentalArguments();

        //"AODE", "AODEsr", "BayesianLogisticRegression", "ComplementNaiveBayes", "DMNBtext", "NaiveBayesSimple", "NaiveBayesUpdateable

        Classifier[] cls= new Classifier[2];

        String[] names = {"NaiveBayes","BayesNet"};

        NaiveBayes naive = new NaiveBayes();
        BayesNet net = new BayesNet();
//        AODE aode = new AODE();
//        AODEsr star = new AODEsr();
//        BayesianLogisticRegression logreg = new BayesianLogisticRegression();
//        ComplementNaiveBayes complement = new ComplementNaiveBayes();
//        DMNBtext dmnb = new DMNBtext();
//        NaiveBayesSimple simple = new NaiveBayesSimple();
//        NaiveBayesUpdateable updateable = new NaiveBayesUpdateable();

        // allocate classifiers
        cls[0]=naive;
        cls[1]=net;
//        cls[3]=star;
//        cls[4]=logreg;
//        cls[5]=complement;
//        cls[6]=dmnb;
//        cls[7]=simple;
//        cls[8]=updateable;

        expSettings.dataReadLocation = basePath;
        expSettings.resultsWriteLocation = "C:/Work/GitHub/tsml/Results/";
        expSettings.forceEvaluation = true; // Overwrite existing results?
        expSettings.debug = true;

        //If splits are not defined, can set here, the default is 50/50 splits
        DatasetLoading.setProportionKeptForTraining(0.5);

        for(int i=0;i<cls.length;i++) {
            expSettings.classifierName = names[i];
            expSettings.classifier = cls[i];
            for (String str : allProblems) {
                // 5 folds
                for (int j = 0; j < 5; j++) {
                    expSettings.datasetName = str;
                    expSettings.foldId = j;  // note that since we're now setting the fold directly, we can resume zero-indexing
                    Experiments.setupAndRunExperiment(expSettings);
                    expSettings.run();
                }
            }
        }
    }


    public static void multipleClassifierEvaluation() throws Exception {
    }
    public static void main(String[] args) throws Exception {
/** PLAN
 * Part 1: Understanding assessment measures
 *  Generate results file
 *  Open in excel
 *  Work out accuracy, TPR etc
 *  Work out NLL
 *  Work out AUROC
 */
        clear();
        // runExperimentManually();
        runMultipleExperiments();

    }

    private static void compareClassifiers() {
    }
}
