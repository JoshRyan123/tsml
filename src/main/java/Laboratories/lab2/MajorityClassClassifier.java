package Laboratories.lab2;

import tsml.src.main.java.experiments.data.DatasetLoading;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class MajorityClassClassifier extends AbstractClassifier {
    // count for training data for how many there are of each class
    int[] count;
    // we have an attribute we are trying to predict
    double[] classDistribution;

    @Override
    public void buildClassifier(Instances data) throws Exception {
        count = new int[data.numClasses()];

        for (Instance ins : data) {
            // returns double so cast to int
            int c = (int) ins.classValue();
            count[c]++;
        }

        classDistribution = new double[data.numClasses()];
        for (int i = 0; i < data.numClasses(); i++)
            classDistribution[i] = count[i] / (double) data.numInstances();
    }

    @Override
    public double[] distributionForInstance(Instance ins) {
        return classDistribution;
    }

    @Override
    public String toString() {
        String str = "Class Distribution  = ";

        for (double d : classDistribution)
            str += d + ",";

        return str;
    }

    public static void main(String[] args) throws Exception{
        Instances all;
        String dataPath = "Data/lab1/Arsenal_TEST.arff";
        all = DatasetLoading.loadData(dataPath);
        //Build on all the iris data
        MajorityClassClassifier mc = new MajorityClassClassifier();
        mc.buildClassifier(all);
        System.out.println("MODEL = " + mc.toString());

        int correct = 0;
        for (Instance ins : all) {
            int pred = (int) mc.classifyInstance(ins);
            int actual = (int) ins.classValue();

            if (pred == actual)
                correct++;
        }
        System.out.println(" num correct = " + correct + " accuracy  = " + (correct / (double) all.numInstances()));
    }

}