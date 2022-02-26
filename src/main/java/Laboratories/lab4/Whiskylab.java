package Laboratories.lab4;

import evaluation.storage.ClassifierResults;
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

import static experiments.TonyCollateResults.collate;
import static org.apache.log4j.NDC.clear;

public class Whiskylab {
    public static String basePath="C:\\Work\\GitHub\\tsml\\Data\\Labs\\";

    static String[] allClassifiers={"NaiveBayes","BayesNet", "AODE", "AODEsr", "BayesianLogisticRegression", "ComplementNaiveBayes", "DMNBtext", "NaiveBayesSimple", "NaiveBayesUpdateable"};
    static String[] allProblems ={"WhiskyData", "Aedes"};

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


        OutFile out = new OutFile("C:/Work/GitHub/tsml/Results/Naive_"+problem+"_Resample0.csv");
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

        Classifier[] cls= new Classifier[5];

        String[] names = {"NaiveBayes","BayesNet", "BayesianLogisticRegression", "DMNBtext", "NaiveBayesUpdateable"};

        // do work:
        NaiveBayes naive = new NaiveBayes();
        BayesNet net = new BayesNet();
        BayesianLogisticRegression logreg = new BayesianLogisticRegression();
        DMNBtext dmnb = new DMNBtext();
//        NaiveBayesSimple simple = new NaiveBayesSimple();
//        System.out.println(simple.getCapabilities());
        NaiveBayesUpdateable updateable = new NaiveBayesUpdateable();

        // dont work:
//        AODE aode = new AODE();
//        System.out.println(aode.getCapabilities());
//        AODEsr star = new AODEsr();
//        System.out.println(star.getCapabilities());
//        ComplementNaiveBayes complement = new ComplementNaiveBayes();
//        System.out.println(complement.getCapabilities());

        // allocate classifiers
        cls[0]=naive;
        cls[1]=net;
        cls[2]=logreg;
        cls[3]=dmnb;
//        cls[4]=simple;
        cls[4]=updateable;

        expSettings.dataReadLocation = basePath;
        expSettings.resultsWriteLocation = "C:/Work/GitHub/tsml/Results/";
        //expSettings.forceEvaluation = true; // Overwrite existing results?
        expSettings.debug = true;

        //If splits are not defined, can set here, the default is 50/50 splits
        DatasetLoading.setProportionKeptForTraining(0.7);

        for(int i=0;i<cls.length;i++) {
            expSettings.classifierName = names[i];
            expSettings.classifier = cls[i];
            for (String str : allProblems) {
                // 10 folds of cross validation
                for (int j = 0; j < 11; j++) {
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

        runExperimentManually();
        runMultipleExperiments();



//        String[] str={
//                "E:\\ResultsDirectory\\",
//                "Z:\\DataDirectory\\",
//                "10",//Number of resamples for each dataset
//                "false",                            //Don’t worry at the moment
//                "ClassifierName",
//                "0"}; //Don’t worry at the moment
//        collate(str);

        // so for the above data I would use
        String[] str={

                "C:/Work/GitHub/tsml/Results/",
                "C:/Work/GitHub/tsml/Data/Labs/",
                "11", // Number of resamples for each dataset
                "false",
                "NaiveBayes",
                "0"};

        collate(str);








        // classify results:

        ClassifierResults res=new ClassifierResults();

        res.setClassifierName("testClassifier");

        res.setDatasetName("C:/Work/GitHub/tsml/Results/Naive_WhiskyData_Resample0.csv");

        res.setBuildTime(2);
        res.setTestTime(1);

        Random rng = new Random(0);
        for (int i = 0; i < 10; i++) { //obvs dists dont make much sense, not important here
            res.addPrediction(rng.nextInt(2), new double[] { rng.nextDouble(), rng.nextDouble()}, rng.nextInt(2), rng.nextInt(5)+1, "test,again");
        }

        res.finaliseResults();

        System.out.println(res.writeFullResultsToString());
        System.out.println("\n\n");

        res.writeFullResultsToFile("C:/Work/GitHub/tsml/Results/Naive_WhiskyData_ClassifyResults.csv");

        ClassifierResults res2 = new ClassifierResults("C:/Work/GitHub/tsml/Results/Naive_WhiskyData_ClassifyResults.csv");
        System.out.println(res2.writeFullResultsToString());

//        res.findAllStats();
//        res.findAllStatsOnce();
//        //res.findBalancedAcc();
//        res.findEarliness();
//        //res.findF1();
//        res.findMeanAUROC();
//        res.findNLL();
//        //res.computeMCC();
//        res.findHarmonicMean();
//        res.allPerformanceMetricsToString();

    }

    private static void compareClassifiers() {
    }
}
