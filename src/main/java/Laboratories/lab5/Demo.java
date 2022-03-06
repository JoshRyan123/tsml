package Laboratories.lab5;

import experiments.data.DatasetLoading;
import org.checkerframework.checker.units.qual.A;
import utilities.InstanceTools;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.lazy.IB1;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.core.matrix.LinearRegression;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class Demo {
    public static String basePath="C:\\Work\\GitHub\\tsml\\Data\\";

    public static void main(String[] args) throws Exception {
//        // write your code here
//        String trainDataLocation="Arsenal_TRAIN.arff";
//        Instances trainData = loadData2(trainDataLocation);
//
//        String testDataLocation="Arsenal_TEST.arff";
//        Instances testData = loadData2(testDataLocation);


        String problem = "FootballPlayers";

        Instances data = DatasetLoading.loadData(basePath + problem + ".arff");

        Instances[] split = splitData(data, 0.7);

        if (split[0].classIndex() == -1)
            split[0].setClassIndex(split[0].numAttributes() - 1);
        if (split[1].classIndex() == -1)
            split[1].setClassIndex(split[1].numAttributes() - 1);

        System.out.println("Number of attributes:"+split[0].numAttributes());
        System.out.println("Number of instances:"+split[0].numInstances());
        System.out.println("Number of class labels:"+split[0].numClasses());
        System.out.println("Class distribution:"+ Arrays.toString(classDistribution(split[0])));

        //oneNN model = new oneNN();
        kNN model = new kNN();
        //weka 1NN:
        //IB1 model = new IB1();
        //weka KNN:
        //IBk model = new IBk();

        model.buildClassifier(split[0]);

        System.out.println("Prediction accuracy:"+accuracy(model, split[1]));

        //remove catagorical data:
        ArrayList<Integer> index = new ArrayList<>();
        for(int k=0; k<data.numAttributes()-1;k++){
            if(data.attribute(k).isNominal()){
                index.add(k);
            }
        }
        int[] index_array = new int[index.size()];
        for(int k=0; k<index_array.length;k++){
            index_array[k] = index.get(k).intValue();
        }
        // removes first 4 attributes from FootballPlayers as they are categorical
        Remove removeFiler = new Remove();
        removeFiler.setAttributeIndicesArray(index_array);
        removeFiler.setInputFormat(data);
        //removeFiler.setInvertSelection(true);

        Instances newData = Filter.useFilter(data, removeFiler);
        Instances[] dividedDataset = splitData(newData, 0.7);
        if (dividedDataset[0].classIndex() == -1)
            dividedDataset[0].setClassIndex(dividedDataset[0].numAttributes() - 1);
        if (dividedDataset[1].classIndex() == -1)
            dividedDataset[1].setClassIndex(dividedDataset[1].numAttributes() - 1);

        kNN model1 = new kNN();
        model1.buildClassifier(dividedDataset[0]);

        System.out.println("Prediction accuracy:"+accuracy(model1, dividedDataset[1]));

        //System.out.println("The trained classifier is: " + model);
        for (Instance inst : dividedDataset[1]) {
            double predResult = model1.classifyInstance(inst);
            double[] predProbResult = model1.distributionForInstance(inst);

            System.out.println("Actual: " + inst.classValue() + "; Predicted: " + predResult);
            System.out.println("Predicted (probability): " + Arrays.toString(predProbResult));
        }
        //sout("prediction accruacy": + accuracy(model, dividedDataset[1]))

        //need to have data dividedDataset for evaluation
        Evaluation evaluation = new Evaluation(dividedDataset[0]);
        evaluation.evaluateModel(model1, dividedDataset[1]);
        System.out.println(evaluation.toSummaryString());
        System.out.println(evaluation.toMatrixString());
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

        for(Instance inst:data){
            index = (int) inst.classValue();
            count[index]++;
        }
        //count appearance divided by total num instances
        for(int k = 0; k < data.numClasses(); k++){
            distribution[k] = ((double) count[k])/data.numInstances();
        }

        return distribution;
    }

    public static Instances[] splitData(Instances data, double proportion) throws Exception{
        int index = (int) (proportion * data.numInstances());
        int seed = 1000;
        Random rand = new Random(seed);
        Instances[] split = new Instances[2];

        //cross validation:
        //Instances randomData = new Instances(data);
        //randomData.randomize(rand);
        //split[0] = randomData.trainCV(2,0,rand);
        //split[1] = randomData.testCV(2,0);

        split[0] = new Instances(data);
        split[0].randomize(rand);
        split[1] = new Instances(data, 0);
        for(int k = data.numInstances()-1; k >= index; k--){
            split[1].add(split[0].instance(k));
            split[0].delete(k);
        }

        return split;
    }

    public static double accuracy(Classifier model, Instances test) throws Exception {
        int prediction, actual;
        int count=0;
        for(Instance inst:test){
            prediction=(int) model.classifyInstance(inst);
            actual = (int) inst.classValue();
            // if prediction and actual are the same increment count
            if(prediction==actual){
                count++;
            }
        }
        return count/(double)test.numInstances();
    }


}
