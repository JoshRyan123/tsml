package Laboratories.lab3;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Random;

// ensemble simulated asking a wide audience for a prediction rather than just consulting one person
public class JoshEnsemble extends AbstractClassifier {
    // Majority vote of j48 classifiers
    // an ensamble is a array of classifiers
    ArrayList<Classifier> ensemble;
    //no. in ensemble?
    // would also need setter methods

    // 500 to compare with RandomForest which defaults 500
    int numClassifiers=500;
    // you might even want a base classifier so that you can change it, would need another method with cloning
    Classifier j48 = new J48();

    /**
     * build an ensemble from scratch, without using any built in
     * tools. Implement an ensemble classifier that contains an array of J48 base
     * classifiers. Diversify your ensemble by sampling 50% of the train data for each
     * classifier (without replacement). Classify new instances with a simple majority
     *
     * @param data set of instances serving as training data
     * @throws Exception
     */
    @Override
    public void buildClassifier(Instances data) throws Exception {

        ensemble = new ArrayList<>(numClassifiers);

        for(int i=0;i<numClassifiers;i++){
            Classifier c = new J48();
            // some classifiers can make random choices from using the same data, if you want to mae it deterministic,
            // (you build it twice) you build it with the same seed.
            // in this case all classifiers have a different seed
            // it were heavily randomized internally mightbe enough to diversify
            ((J48)c).setSeed(i);
            c.buildClassifier(data);
            ensemble.add(c);
        }

        // diversifying via sampling without replacement (different from bagging)
        // in bagging we sample with replacement

        // the ensemble is homogenious cause all classifiers are the same type (also is not iterative, as is more harder,
        // and involves the use of gradient boosting: e.g. adaboost, logiboost)
        // involves analysing the training data and where mistakes in prediction aride the code should pass this
        // onto future classifiers to improve prediction (done by giving more importance or weighting depending on the
        // incorrect prediction)
        // weka.classifiers.meta.AdaBoostM1
        // also look into XGBoosting as is a more modern alternative and might be easier to implement?

        // problem is j48 is deterministic meaning that if we have 2 of them, they will respond with the same result,
        // making any scalable amount of them redundant
        //
        // 2 methods for diversifying:
        // change the data (standard), the following is just one way of doing this!!!
        // change the classifier
        for(Classifier c:ensemble){
            //Split the data: use tools or do it manually, gives classifiers different subset of the training data
            data.randomize(new Random());
            // taking first half of randomized data for the training data (data, index_first, index_last)
            // there is another Instances() constrictor that allows us to copy from a proportion of the data - there
            // are other ways of doing this
            // best outcome is to have a around 70% split but overload the original dataset with data
            // to nullify the amount of data lost by splitting and losing 30%
            Instances train=new Instances(data,0,data.numInstances()/2);

            // could have implemented bagging here <-----------

            //build model on the above
            c.buildClassifier(train);

            /*the point of diversifying is not to get the most highly optimized model possible, but to improve the
            model in general through use of good experimental technique and understanding what you are doing
             */
        }
    }
    // if your creating a distribution: then classifyInstance will just be the argMax of that distribution
    public double classifyInstance(Instance inst) throws Exception{
        // Majority vote of j48 classifiers
        // 1. ask each base classifier (ensemble members) for prediction
        // 2. count how many predict for each class
        // 3. return the one with the highest count (most votes)

        // count for each class
        int[] votes=new int[inst.numClasses()];

        // count predictions for each class
        for(Classifier c:ensemble){
            //cast to int as classifyInstance returns double
            votes[(int)c.classifyInstance(inst)]++;
        }

        // return the one with the highest count (most votes) (need to find argMax)
        // dont want to find highest number of votes!
        // wanna find the index of the highest number of votes as this will be the class value we predict
        // so we will go through 0, 1, 2; check the number of votes and then record 0, 1, 2 to return
        int argMax=0;
        for(int i=1;i<votes.length;i++)
            if(votes[i]>votes[argMax])
                argMax=i;
        return argMax;
    }

    //
    public double[] distributionForInstance(Instance inst) throws Exception {
        // average the probabilities rather than count the votes
        double[] probs= new double[inst.numClasses()];
        // for classifier c gets the probability estimate of each class value
        for(Classifier c:ensemble){
            //cast to int as classifyInstance returns double
            double[] d = c.distributionForInstance(inst);
            for (int i=0; i<d.length; i++) {
                probs[i]+=d[i];
            }
        }
        double sum = 0;
        for(int i = 0; i<probs.length; i++){
            sum+=probs[i];
        }
        for(int i = 0;i<probs.length; i++){
            probs[i]/=sum;
        }
        return probs;
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = new Capabilities(this);
        result.disableAll();
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.BINARY_CLASS);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);

        return result;
        // copying J48 capabilities
//        Capabilities result = new J48().getCapabilities();
//        return result;
    }
}
// discuession on various boosting methods held at the end [1:02:00]
