package Laboratories.lab5;

//regression problems come more in data science and involve using the intercept and slope of a line of regression
//to calculate and predict values for contiguous data such as temperature, time.
//classification problems come more in machine leaning and involve classification using a metric (for 1NN we used distance metric
//which used the sum of square error) in order to get a prediction and subsequent distribution.

//discrete vs continuous variables

import de.bwaldvogel.liblinear.Train;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;
import java.util.Comparator;

//nearest neighbour classifier
//needs to extend AbstractClassifier
public class kNN extends AbstractClassifier {
    //global variable
    Instances TrainData;
    int num_k = 90;

    //nearest neighbour is a lazy classifier; it does nothing in the training stage
    //assigns TrainData variable to the data
    @Override
    public void buildClassifier(Instances insts) throws Exception {
        //no learning is happening so we just store the data
        TrainData = insts;
    }

    //where the projection happens - returns a prediction (double)
    //prediction stage;
    @Override
    public double classifyInstance (Instance inst) throws Exception{
        //want to predict class of new instance observed
        //predict the new instance as a class belonging to its nearest neighbour

        //initialised to very large positive value (starting point) (dont need)
//        double NN_dist = Double.POSITIVE_INFINITY;

        //assigned to any value as is updated afterwards (dont need)
//        Instance NN_inst = TrainData.firstInstance();

        //need to calculate multiple distances
        double[] dist = new double[TrainData.numInstances()];

        int k = 0;

        //look through training dataset; for each; calculate distance
        for (Instance obs:TrainData)
        {
            //calculate distance; (between new instance "inst" and observed instance "obs")
            //this is because NN is measured by a similarity metric (lots of these)

            // store in array
            dist[k] = distance(obs, inst);
            k++;
        }
        //sorting: finding index of nearest neighbours
        Integer[] indices = sortIndex(dist);

        //find classes belonging to the nearest indices
        double[] votes = new double[TrainData.numClasses()];

        //majority vote of nearest neighbours
        for(int i = 0; i<num_k; i++){
            //find class of nn and vote
            votes[(int) TrainData.instance(indices[i]).classValue()] += 1;
        }

        Integer[] indices_vote = sortIndex(votes);

        // predicted class for instance
        return (double) indices_vote[votes.length-1];

    }

    //for NN the class predicted has 100% distribution
    @Override
    public double[] distributionForInstance(Instance inst) throws Exception{

        //need to calculate multiple distances
        double[] prob = new double[TrainData.numInstances()];

        int k = 0;

        //look through training dataset; for each; calculate distance
        for (Instance obs:TrainData)
        {
            //calculate distance; (between new instance "inst" and observed instance "obs")
            //this is because NN is measured by a similarity metric (lots of these)

            // store in array
            prob[k] = distance(obs, inst);
            k++;
        }
        //sorting: finding index of nearest neighbours
        Integer[] indices = sortIndex(prob);

        //find classes belonging to the nearest indices
        double[] votes = new double[TrainData.numClasses()];

        //majority vote of nearest neighbours
        for(int i = 0; i<num_k; i++){
            //find class of nn and vote
            votes[(int) TrainData.instance(indices[i]).classValue()] += 1;
        }

        //find classes belonging to the nearest indices
        double[] probabilities = new double[TrainData.numClasses()];

        for(int i = 0; i<votes.length; i++){
            //actual value of votes divided by num of nearest neighbours
            probabilities[i] = votes[i]/num_k;
        }

        return probabilities;
    }

    //distance function
    public double distance(Instance inst1, Instance inst2) {
        //assume multiple input attributes and therefore will need to loop through them
        //when calculating the sum of square error between inst1 and inst2
        double sum = 0;

        //calculate sum of square error for each attribute in training dataset
        //numAttributes()"-1" because dont want to include class/target attribute (it is only calculated using the feature space)
        for(int k=0; k<TrainData.numAttributes()-1; k++)
        {
            //(instance1-instance2)**2
            sum += Math.pow(inst1.value(k)-inst2.value(k), 2);
        }

        return Math.sqrt(sum);
    }

    public Integer[] sortIndex(double[] arr){
        //initialise indicies
        Integer[] indicies = new Integer[arr.length];
        for (int i = 0; i < indicies.length; i++){
            indicies[i] = i;
        }

        Arrays.sort(indicies, new Comparator<Integer>() {
            public int compare(Integer i1, Integer i2) {
                return Double.compare(arr[i1],arr[i2]);
            }
        });

        return indicies;
    }

    @Override
    public String toString() {
        return "oneNN{" +
                "TrainData=" + TrainData +
                '}';
    }
}