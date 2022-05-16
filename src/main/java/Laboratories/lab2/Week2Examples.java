package Laboratories.lab2;

import Laboratories.lab1.HistogramClassifier;
import experiments.data.DatasetLoading;
import utilities.InstanceTools;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * Week 2 demo todo:
 *
 * 1. Run through majority class classifier with iris and a discrete problem
 *      1.1 fit and predict on train
 *      1.2 Do first train/test split.
 *      1.3 Introduce Evaluation https://weka.sourceforge.io/doc.dev/index.html?weka/classifiers/Evaluation.html
 *
 * 2. Run through histogram classifier
 *
 * 3. Do an information gain example by hand
 *
 * 4. Measure info gain in code (see lab sheet)
 *
 * 5. Build a tree by hand
 *
 * 6. Build an ID3 Tree in code
 *
 */

public class Week2Examples {
    public static void simpleDemo(Classifier cls, Instances data) throws Exception {
        // build model using training data
        cls.buildClassifier(data);
        System.out.println(cls);

        int correct=0;

        //
        for(Instance ins:data) {
            int pred = (int) cls.classifyInstance(ins);
            int actual = (int) ins.classValue();
//            System.out.println(" Predicted  = "+pred);
            if (pred == actual)
                correct++;
//            double[] probs = mc.distributionForInstance(ins);
        }
        //System.out.println(" Train set correct = "+correct+" Accuracy = "+(correct/(double)data.numInstances()));

        // splits data into train and test 50% train 50% test
        Instances[] split = InstanceTools.resampleInstances(data,0,0.5);

        // Provides way of setting up classifier and a training set and then assessing how good the classifier is on
        // the training set
        Evaluation evaluation = new Evaluation(split[0]);
        evaluation.evaluateModel(cls, split[1]);

        //System.out.println(" Evaluation  = "+evaluation.toSummaryString());
    }

    public static void discreteProblem() throws Exception {
        Instances train = DatasetLoading.loadData("Data\\Arsenal_TRAIN.arff");
        Instances test = DatasetLoading.loadData("Data\\Arsenal_TEST.arff");

        // loops through different attributes 'k'
        for(int k=0;k<train.numAttributes()-1;k++) {
            // num values for given attribute k
            int numVals = train.attribute(k).numValues();
            // rows are the attribute values (no. of possible values for attribute) and columns are class values (outcome).
            int[][] counts = new int[numVals][train.numClasses()];
            // gives count
            for (Instance ins : train) {
                // use class and att to index into counts
                int cls = (int) ins.classValue();
                int att = (int) ins.value(k);
                counts[att][cls]=counts[att][cls]+1;
            }
            for (int i = 0; i < counts.length; i++) {
                if(i==0){
                    //System.out.println("Player results when not playing:");
                }
                else if(i==1){
                    //System.out.println("Player results when playing:");
                }
                for (int j = 0; j < counts[i].length; j++)
                    System.out.print(counts[i][j] + ",");
                //System.out.println("");
            }

            //Find probs

            //Find the info gain

        }

    }

    public static void main(String[] args) throws Exception {
        Instances wdbc = DatasetLoading.loadData("Data\\Labs\\Aedes\\Aedes.arff");
        Instances afc = DatasetLoading.loadData("Data\\Arsenal_TRAIN.arff");

        // using Arsenal_TRAIN.arff data and MajorityClassClassifier
        MajorityClassClassifier mc= new MajorityClassClassifier();
        simpleDemo(mc,afc);

        // using Aedes_Female_VS_House_Fly_POWER.arff data and HistogramClassifier
        HistogramClassifier hc = new HistogramClassifier();
        simpleDemo(hc, wdbc);

        discreteProblem();
    }

}
