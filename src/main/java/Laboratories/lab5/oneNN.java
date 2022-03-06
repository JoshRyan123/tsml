package Laboratories.lab5;

//regression problems come more in data science and involve using the intercept and slope of a line of regression
//to calculate and predict values for contiguous data such as temperature, time.
//classification problems come more in machine leaning and involve classification using a metric (for 1NN we used distance metric
//which used the sum of square error) in order to get a prediction and subsequent distribution. Examples of discrete class values
//include a cat or dog or a burch tree or oak tree

//classification=discrete
//regression=continuous

//discrete vs continuous variables

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

//nearest neighbour classifier
//needs to extend AbstractClassifier
public class oneNN extends AbstractClassifier {
    //global variable
    Instances TrainData;

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

        //initialised to very large positive value (starting point)
        double NN_dist = Double.POSITIVE_INFINITY;

        //assigned to any value as is updated afterwards
        Instance NN_inst = TrainData.firstInstance();

        //need to calculate distance
        double dist;

        //look through training dataset; for each; calculate distance
        for (Instance obs:TrainData)
        {
            //calculate distance; (between new instance "inst" and observed instance "obs")
            //this is because NN is measured by a similarity metric (lots of these)
            dist = distance(obs, inst);

            //find smallest distance between new instance and observed instances
            if(dist < NN_dist)
            {
                //update smallest distance
                NN_dist = dist;
                //assign/update instance with smaller distance than previously observed
                NN_inst = obs;
            }
        }

        //return class value of instance with the smallest distance
        return NN_inst.classValue();

    }

    //for NN the class predicted has 100% distribution
    @Override
    public double[] distributionForInstance(Instance inst) throws Exception{
        //array of equal length to number of classes (returns probability of selecting each possible class value for an instance)
        double[] predProb = new double[TrainData.numClasses()];

        double pred = classifyInstance(inst);

        //set the probability for the class value we predicted to be 100%; rest of classes kept at 0%
        predProb[(int) pred] = 1.0;

        return predProb;
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

    @Override
    public String toString() {
        return "oneNN{" +
                "TrainData=" + TrainData +
                '}';
    }
}