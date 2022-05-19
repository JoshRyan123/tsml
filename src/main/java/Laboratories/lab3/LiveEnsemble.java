package Laboratories.lab3;

import experiments.data.DatasetLoading;
import utilities.InstanceTools;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;

public class LiveEnsemble {
    public static void main(String[] argv) throws Exception {
        String dataPath = "Data/lab1/Aedes_Female_VS_House_Fly_POWER.arff";
        Instances all = DatasetLoading.loadData(dataPath);
        all.setClassIndex(all.numAttributes() - 1);

        //splitting data: (split[0] will store training data and split[1] will stores test data)
        Instances[] split = InstanceTools.resampleInstances(all, 0, 0.5);

//
        JoshEnsemble ens = new JoshEnsemble();
        ens.buildClassifier(split[0]);
//        System.out.println(" JoshEnsemble capabilities = " + ens.getCapabilities());
//
        J48 c45 = new J48();
        c45.buildClassifier(split[0]);

        // making sure distributionForInstance works
        // if you want to test two classifiers you don't want to do this on the dataset you build them on: why because
        // you've optimized your algorithm on that data which gives bias results.


        // time for bagging
        JoshBagging bagging = new JoshBagging();
        bagging.setClassifier(new J48());

        bagging.buildClassifier(split[0]);

        // going more than 10 breaks
        bagging.setNumIterations(10);

        // A random seed is a starting point in generating random numbers
        System.out.println("random seed=\n"+bagging.getSeed());

        //key part: for each classifier the training data is resampled without replacement (bootstraping) (resamples the data)
        bagging.setBagSizePercent(500);

        System.out.println("(Bagging)" + bagging.getCapabilities());


        // time for random forrest with the bags
        RandomForest randForest = new RandomForest();
        // set number of threads
        randForest.setNumExecutionSlots(5);

        randForest.setNumTrees(10);
        randForest.setSeed(1);
        //randForest.setCalcOutOfBag(true);
        randForest.buildClassifier(split[0]);


        randForest.setNumTrees(500);


        // additionally should check out adaboost
        int countC45 = 0;
        int countBagging = 0;
        int countEns = 0;
        int countForest = 0;

        for(Instance inst: split[0]){
            int pred = (int)c45.classifyInstance(inst);
            int pred2 = (int)bagging.classifyInstance(inst);
            int pred3 = (int)ens.classifyInstance(inst);
            int pred4 = (int)randForest.classifyInstance(inst);
            int actual = (int)inst.classValue();
            System.out.println(" Actual = "+actual+"\n Bag Predicted ="+pred2);
            System.out.println(" c45 Predicted ="+pred);
            System.out.println(" Ens Predicted ="+pred3+"\n");
            System.out.println(" RandomForest Predicted ="+pred4+"\n");

            if(pred==actual){
                countC45++;
            }
            if(pred2==actual){
                countBagging++;
            }
            if(pred3==actual){
                countEns++;
            }
            if(pred4==actual){
                countForest++;
            }
        }
        System.out.println(" c45 correct = " + countC45 + " c45 accuracy  = " + (countC45 / (double) all.numInstances()));
        System.out.println(" c45 bagging num correct = " + countBagging + " c45 bagging accuracy  = " + (countBagging / (double) all.numInstances()));
        System.out.println(" 500 ensemble num correct = " + countEns + " 500 ensemble accuracy  = " + (countEns / (double) all.numInstances()));
        System.out.println(" 500 ensemble num correct = " + countForest + " 500 ensemble accuracy  = " + (countForest / (double) all.numInstances()));

        // BinaryRelevance BR = new BinaryRelevance(bagging);

        //System.out.println(bagging.toString());
        System.out.println(randForest.getNumFeatures());
    }
}
