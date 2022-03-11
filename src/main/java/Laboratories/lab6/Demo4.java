package Laboratories.lab6;

import Laboratories.lab5.kNN;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.FileReader;
import java.util.Arrays;
import java.util.Random;

public class Demo4 {

    public static void main(String[] args) throws Exception {

        String dataLocation="FootballPlayers.arff";
        Instances data = loadData2(dataLocation);

        if(data.classIndex() == -1)
            data.setClassIndex(data.numAttributes()-1);

        System.out.println("Number of attributes:" + data.numAttributes());
        System.out.println("Number of instances:" + data.numInstances());
        System.out.println("Number of class labels:" + data.numClasses());
        System.out.println("Class distribution:" + Arrays.toString(classDistribution(data)));

        Instances[] dividedDataSet = splitData(data, 0.7);

        kNN model = new kNN();
        model.buildClassifier(dividedDataSet[0]); // training

        double[] predProbResult = model.distributionForInstance(dividedDataSet[1].instance(0));
        System.out.println("Predicted (posterior probability): " + Arrays.toString(predProbResult));

        // System.out.println("Prediction accuracy:" + accuray(model, dividedDataSet[1])); // testing
    }

    public static Instances loadData1(String dataLocation){
        Instances data = null;
        try{
            FileReader reader = new FileReader(dataLocation);
            data = new Instances(reader);
        }catch(Exception e){
            System.out.println("Exception caught: "+e);
        }
        return data;
    }

    public static Instances loadData2(String dataLocation){
        Instances data = null;
        try{
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(dataLocation);
            data = source.getDataSet();
        }catch(Exception e){
            System.out.println("Exception caught: "+e);
        }
        return data;
    }

    public static double[] classDistribution(Instances data){
        int[] count = new int[data.numClasses()];
        double[] distribution = new double[data.numClasses()];
        int index;

        for(Instance inst:data)
        {
            index = (int) inst.classValue();
            count[index]++;
        }

        for(int k = 0; k < data.numClasses(); k++)
        {
            distribution[k] = ((double) count[k])/data.numInstances();
        }

        return distribution;
    }

    public static double accuray(Classifier model, Instances test) throws Exception {

        int prediction, actual;
        int count = 0;
        for(Instance inst:test)
        {
            prediction = (int) model.classifyInstance(inst);
            actual = (int) inst.classValue();
            if(prediction == actual)
                count++;
        }

        return count/(double) test.numInstances();

    }

    public static Instances[] splitData(Instances data, double proportion) throws Exception {

        int index = (int) (proportion * data.numInstances());

        int seed = 1000;
        Random rand = new Random(seed); // get the number generator with a seed

        Instances[] split = new Instances[2];

        split[0] = new Instances(data);
        split[0].randomize(rand);

        split[1] = new Instances(data, 0);

        for(int k = data.numInstances()-1; k >= index; k--){
            split[1].add(split[0].instance(k));
            split[0].delete(k);
        }

        return split;
    }
}
