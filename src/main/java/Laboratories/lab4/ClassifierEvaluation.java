package Laboratories.lab4;

import evaluation.MultipleClassifierEvaluation;
import evaluation.tuning.ParameterSpace;
import experiments.Experiments;
import experiments.data.DatasetLoading;
import fileIO.OutFile;
import machine_learning.classifiers.tuned.TunedClassifier;
import utilities.InstanceTools;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.*;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Random;

public class ClassifierEvaluation {

    public static String basePath="C:\\Work\\GitHub\\tsml\\Data\\UCI\\";

    public static void tunedComparison(){

    }

    static String[] allClassifiers={"J48","SimpleCart","FT","HoeffdingTree","LADTree","REPTree","DecisionStump","J48graft"};
    static String[] allProblems ={"blood", "hayes-roth","bank","balance-scale","monks-1","breast-cancer-wisc-diag"};

    public static Classifier setClassifier(String c){
        switch(c){
            case "J48":
                return new J48();
            case"SimpleCart":
                return new SimpleCart();
            case "FT":
                return new FT();
            case "HoeffdingTree":
                return new HoeffdingTree();
            case "LADTree":
                return new LADTree();
            case "NBTree":
                return new NBTree();
            case "REPTree":
                return new REPTree();
            case "DecisionStump":
                return new DecisionStump();
            case "J48graft":
                return new J48graft();
        }
        return null;

    }//,"trains"

    public static boolean isPig(String str){
        switch(str){
            case "connect-4":
            case "statlog-shuttle":
            case "chess-krvk":
            case "nursery":
            case "ringnorm":
            case "twonorm":
            case "musk-2":
            case "statlog-landsat":
            case "optical":
            case "page-blocks":
            case "wall-following":
            case "waveform-noise":
            case "bank":
                return true;
        }
        return false;

    }

    public static void runExperimentManually() throws Exception {

        String problem = "optical";
        Instances data = DatasetLoading.loadData(basePath+problem+"/"+problem+".arff");

        System.out.println(data.numClasses());

        Instances[] split = InstanceTools.resampleInstances(data, 0, 0.5);
        //
        //split = DatasetLoading.sampleHayesRoth(0);

        Classifier c = new J48();
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
        Evaluation eval = new Evaluation(split[0]);

        // 1 model:
        eval.evaluateModel(c, split[1]);
        // 10 models:
        eval.crossValidateModel(c, data, 10, new Random());

        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
    }


    public static void runExperimentAutomatically() throws Exception {
        Experiments.ExperimentalArguments expSettings = new Experiments.ExperimentalArguments();
        TunedClassifier tuned = new TunedClassifier();
        expSettings.classifier = new J48();
        expSettings.dataReadLocation = basePath;
        expSettings.resultsWriteLocation = "C:/Work/GitHub/tsml/Results/";
        expSettings.classifierName = "C45";
        expSettings.datasetName = "bank";
        expSettings.forceEvaluation = true; // Overwrite existing results?
        expSettings.foldId = 10;  // note that since we're now setting the fold directly, we can resume zero-indexing
        expSettings.debug = true;

        //If splits are not defined, can set here, the default is 50/50 splits
        DatasetLoading.setProportionKeptForTraining(0.75);
        Experiments.setupAndRunExperiment(expSettings);
        expSettings.run();

        expSettings.classifierName="TunedC45";
        expSettings.classifier=tuned;
        expSettings.forceEvaluation=false;
        expSettings.run();
        System.out.println(" TunedC45 = " + tuned.getParameters());
    }


    public static void evaluateInCode() throws Exception {
        String problem = "bank";
        Instances data = DatasetLoading.loadData(basePath+problem+"/"+problem+".arff");

        Instances[] split = InstanceTools.resampleInstances(data,0,0.5);

        // evaluation class (weka) can be used to evaluate in many ways described in the week 4 lectures such as TPV
        // and TNV
        Evaluation eval = new Evaluation(split[0]);

        Classifier c = new J48();
        c.buildClassifier(split[0]);

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



    public static void runMultipleExperiments() throws Exception {
        // create expSettings instance for parameter setting
        Experiments.ExperimentalArguments expSettings = new Experiments.ExperimentalArguments();

        Classifier[] cls= new Classifier[4];
        String[] names = {"C45", "RandF","RandF500", "TunedC45"};

        //C45
        J48 c45 = new J48();

        //RandF
        RandomForest randfDefault = new RandomForest();

        //RandF500
        RandomForest randf = new RandomForest();
        randf.setNumTrees(500);

        //Tuned C45
        TunedClassifier tunedC45 = new TunedClassifier();
        tunedC45.setClassifier(new J48());

        //Tune on minimum number of instances, for example
        int[] range = new int[10];
        // i = 1 : range[1] = 4
        // i = 2 : range[1] = 6
        // i = 3 : range[1] = 8
        // i = 4 : range[1] = 10
        // i = 5 : range[1] = 12
        // i = 6 : range[1] = 14
        // ...
        for(int i=0;i<10;i++)
            range[i] = 2+i*2;
        // -M
        ParameterSpace ps = new ParameterSpace();
        ps.addParameter("M",range);
        tunedC45.setParameterSpace(ps);

        // allocate classifiers
        cls[0]=c45;
        cls[1]=randfDefault;
        cls[2]=randf;
        cls[3]=tunedC45;

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
        System.out.println("Classifier evaluation begins");
        // tsml code:
        MultipleClassifierEvaluation mce = new MultipleClassifierEvaluation("C:/Work/GitHub/tsml/Results/","C45vsRandFvsRandF500vsTunedC45", 5);
        // parameters:
        mce.setDatasets(allProblems);
        mce.readInClassifiers(new String[] {"C45","RandF","RandF500","TunedC45"},"C:/Work/GitHub/tsml/Results/");
        mce.setIgnoreMissingResults(true);//remove null
        mce.setBuildMatlabDiagrams(false);
        mce.setTestResultsOnly(true);
        mce.setDebugPrinting(true);
        mce.runComparison();
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
        //    runExperimentManually();
        //    runExperimentAutomatically();
        //    runMultipleExperiments();

        /*  Part 2: Generating performance measures in code */
        //           evaluateInCode();
        /*  Part 3: Generating performance measures from results files */
        //Create results files
        multipleClassifierEvaluation();

        //Compare
        //compareClassifiers();

        //generateResultsExample(30);
        //collateStatsExample(30);
        //createSingleResultsFile(30);
    }

    private static void compareClassifiers() throws Exception {
    }
}
// Creating CM in Excel : [14:00]
// Proportion correct = numCorrect/total on Excel : [14:40]
// Likelihood on Excel = [19:30]
// - leplas correction: hack (if its 0 we set to a very low value otherwise multiplying by a 0 would give a value of 0!)
