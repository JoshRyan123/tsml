package Laboratories.lab4;

import lab3.J48;
import tsml.src.main.java.evaluation.MultipleClassifierEvaluation;
import tsml.src.main.java.evaluation.tuning.ParameterSpace;
import tsml.src.main.java.experiments.data.DatasetLoading;
import tsml.src.main.java.fileIO.OutFile;
import tsml.src.main.java.machine_learning.classifiers.tuned.TunedClassifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;
import java.util.Random;

public class EvaluationExample {
    public static String path ="C:\\Work\\Machine Learning\\ML-Laboratories\\Data\\UCI Continuous\\";

    public static void TunedComparison(){
        J48 c45 = new J48();

        // to tune you must specify a range of parameters
        // cross validation for each parameter combination to get an accuracy/error and chooce the combination that
        // minimises the cross validational error on the training data and maximises accuracy
        // check out J48 method; setOptions()

        // tuning minNumObjects: which is the stopping criterian for a tree (min num instances)
        // default is 2 as usually a tree node only stops once it is pure: meaning having a distribution of 1 or 0
        // i.e one node is all 1 and one node is all 0
        // if this value is changes you could have more than 2 nodes be at the end of a branch

        // tsml james classifier: go through yourself
        TunedClassifier TunedC45 = new TunedClassifier();

        // setup param:
        int size = 10;
        int[] cs = new int[size];
        for (int i = 0; i<cs.length; i++){
            cs[i] = 2 * (i + 1);
        }
        ParameterSpace ps = new ParameterSpace();
        // M - minNumObjects - equivalent to minimum number of instances per leaf
        ps.addParameter("M",cs);
        TunedC45.setClassifier(c45);
        TunedC45.setParameterSpace(ps);


        //run experiment
        //tsml.examples for some example useages tuned classifier setup
        // for each problem construst default c45 J48 classifier and tuned J48 c45 classifier and evaluate on test data for
        // all problem sets
        for (String str:problems) {
            Experiments.ExperimentalArguments expSettings = new Experiments.ExperimentalArguments();
            expSettings.classifierName = "C45";
            expSettings.dataReadLocation = path;
            expSettings.resultsWriteLocation = "C:/Work/Machine Learning/ML-Laboratories/Results/";
            expSettings.datasetName = str;
            // NEEDS FIXING
            //expSettings.classifier
            expSettings.run();

//            expSettings.classifierName = "TunedC45";
            //expSettings.classifier
            expSettings.forceEvaluation = true; // Overwrite existing results?
            // repeated resamples: loop and change fold Id each time
            expSettings.foldId = 10;  // note that since we're now setting the fold directly, we can resume zero-indexing
            expSettings.debug = true;
            expSettings.run();

            //If splits are not defined, can set here, the default is 50/50 splits
            DatasetLoading.setProportionKeptForTraining(0.75);
            //experiments.Experiments.setupAndRunExperiment(expSettings);
            expSettings.run();
        }

    }

    static String[] problems ={
            "blood",
            "bank",
            "breast-tissue",
            "musk-2",
            "breast-cancer-wisc-diag"};

    public static void singleProblemExperiment() throws Exception {
        String str = "breast-cancer-wisc-diag";

        Instances train, test, all;
        String path ="C:\\Work\\Machine Learning\\ML-Laboratories\\Data\\UCI Continuous\\";
        train = DatasetLoading.loadData(path + str + "\\" + str + "_TRAIN.arff");
        test = DatasetLoading.loadData(path + str + "\\" + str + "_TEST.arff");

        File f = new File("C:\\Work\\Machine Learning\\ML-Laboratories\\Results\\C45\\Predictions\\breast-cancer-wisc-diag");

        boolean b = f.mkdirs();

        System.out.println("Mkdirs = "+b);

        OutFile out = new OutFile("C:\\Work\\Machine Learning\\ML-Laboratories\\Results\\C45\\Predictions\\breast-cancer-wisc-diag\\testFold0.csv");
        out.writeLine("J48");
        J48 c45 = new J48();
        out.writeLine("MinNumObjects:"+c45.getMinNumObj());
        c45.buildClassifier(train);

        for(Instance ins:test){
            int pred = (int)c45.classifyInstance(ins);

            double[] probs = c45.distributionForInstance(ins);

            out.writeString((int)ins.classValue()+","+pred+",,");
            for (double d:probs){
                out.writeString(d+",");
            out.writeString("\n");
            }
        }


        // Evaluation module: have to look through this module yourself (lots of useful calculation functions provided)
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(c45,test);
        System.out.println(eval.toSummaryString("\nResults\n========\n", false));


        // cross validation of data with 10 folds
        all = DatasetLoading.loadData(path + str + "\\" + str + ".arff");
        // randomizing is rather important
        // you can speacify a seed for random: random(0): reproduces the same randomization instead of random random seed
        all.randomize(new Random());
        for(int i=0; i<10; i++){
            train = all.trainCV(10, i);
            test = all.testCV(10, i);
            c45=    new J48();
            c45.buildClassifier(train);
//            for(Instance ins:test){
//                int pred = (int)c45.classifyInstance(ins);
//
//                double[] probs = c45.distributionForInstance(ins);
//
//                out.writeString((int)ins.classValue()+","+pred+",,");
//                for (double d:probs){
//                    out.writeString(d+",");
//                    out.writeString("\n");
//                }
//            }
        }
        eval = new Evaluation(train);
        eval.crossValidateModel(c45,all,10,new Random());
        System.out.println(eval.toSummaryString("\nResults\n========\n", false));
        System.out.println("AUC:"+eval.areaUnderROC(0));

    }


    public static void runMultipleExperiments() throws Exception {
        String path ="C:\\Work\\Machine Learning\\ML-Laboratories\\Data\\UCI Continuous\\";
        System.out.println("no. problems ="+problems.length);

        J48 c45 = new J48();
        TunedClassifier TunedC45 = new TunedClassifier();

        // setup param:
        int size = 10;
        int[] cs = new int[size];
        for (int i = 0; i<cs.length; i++){
            cs[i] = 2 * (i + 1);
        }
        ParameterSpace ps = new ParameterSpace();
        ps.addParameter("M",cs);
        TunedC45.setParameterSpace(ps);
        TunedC45.setClassifier(c45);

        for (String str:problems) {
            Experiments.ExperimentalArguments expSettings = new Experiments.ExperimentalArguments();
            expSettings.classifierName = "C45";
            expSettings.dataReadLocation = path;
            expSettings.resultsWriteLocation = "C:/Work/Machine Learning/ML-Laboratories/Results/";
            expSettings.datasetName = str;
            //expSettings.classifier = c45;
            expSettings.run();

//            expSettings.classifierName = "TunedC45";
//            //expSettings.classifier
            expSettings.forceEvaluation = true; // Overwrite existing results?
//            expSettings.foldId = 10;  // note that since we're now setting the fold directly, we can resume zero-indexing
//            expSettings.debug = true;
//            expSettings.run();

            //If splits are not defined, can set here, the default is 50/50 splits
            DatasetLoading.setProportionKeptForTraining(0.75);
            //experiments.Experiments.setupAndRunExperiment(expSettings);
            expSettings.run();
        }
    }

    // collating all the found data and finding out which classifier is best
    public static void multipleClassifierEvaluation() throws Exception {
        System.out.println("Classifier evaluation begins");
        // tsml code:
        MultipleClassifierEvaluation mce = new MultipleClassifierEvaluation("C:/Work/Machine Learning/ML-Laboratories/Results/","C45vsRandF", 5);
        // parameters:
        mce.setDatasets(problems);
        // "RandF500","TunedC45"
        mce.readInClassifiers(new String[] {"C45","RandF"},"C:/Work/Machine Learning/ML-Laboratories/Results/");

        mce.setIgnoreMissingResults(true);//remove null

        //need matlabs
        mce.setBuildMatlabDiagrams(false);
        mce.setTestResultsOnly(true);

        mce.setDebugPrinting(true);

        mce.setUseAllStatistics();

        mce.runComparison();

        //sktime:
        //mathplotlib:
    }

    public static void main(String[] args) throws Exception {
        // using build in weka evaluator
        singleProblemExperiment();
        runMultipleExperiments();
        TunedComparison();

    }
}
