package ml_6002b_coursework;

import Laboratories.lab3.J48;
import Laboratories.lab3.JoshBagging;
import Laboratories.lab3.JoshEnsemble;
import experiments.data.DatasetLoading;
import scala.tools.nsc.transform.patmat.Logic;
import utilities.InstanceTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

/**
 * Ensemble of CourseworkTree classifiers using simply majority vote with option to use average probabilities
 *
 * 1. All with same train data
 * 2. Diversify on train data: sample attributes
 * 3. Stores attribute subsets in
 * 4. Diversify on decision tree parameters: randomize (proportion of attributes too)
 * 5. Ensemble parameter "numTrees": 50
 * 6. Ensemble parameter "proportion of attributes": 50%
 * 7. Loads optdigits and chinetown and print test accuracies and probability estimates
 */
public class TreeEnsemble extends AbstractClassifier{

    int numTrees=50;

    int attributeSelectionSize=20;  // 20%

    // Homogenious array of dt's
    ArrayList<CourseworkTree> treeEnsemble;

    // Store which attributes are used with which classifier in order to recreate the attribute selections
    // in classifyInstance and distributionForInstance
    String[] attributeBags;

    boolean averageDistributions;

    public void setAverageDistributions(boolean average){
        this.averageDistributions = average;
        // System.out.println("Average distributions? : "+average+"\n");
    }


    @Override
    public Capabilities getCapabilities(){
        Capabilities result = new Capabilities(this);
        result.disableAll();
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.BINARY_CLASS);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        return result;
    }

    // construct new sets of instances for base classifiers (random subset of attributes)
    @Override
    public void buildClassifier(Instances data) throws Exception {
        /*
            the point of diversifying is to improve the model in general through use of good experimental technique
            and understanding what you are doing.
        */
        // 2 methods for diversifying:
        // change the classifier
        // change the data (standard)

        treeEnsemble = new ArrayList<>(numTrees);

        // array of [ratio 1, gain 1, chi 1, gini 1] for attribute selection mechanism
        String[] attributeSelectionOptions = new String[4];
        attributeSelectionOptions[0] = "-g 1";
        attributeSelectionOptions[1] = "-i 1";
        attributeSelectionOptions[2] = "-c 1";
        attributeSelectionOptions[3] = "-r 1";
//        for(int i=0;i<attributeSelectionOptions.length;i++) {
//            System.out.println(attributeSelectionOptions[i]);
//        }

        // random number between 1 and 50 for depth
        String[] depthOptions = new String[data.numAttributes()-1];
        Random rn = new Random();
        for (int j = 0; j < depthOptions.length; j++) {
            // Obtain a number between [0 - 49] so increment the result
            int depth = rn.nextInt(5);
            depth++;
            String temp = Integer.toString(depth);
            depthOptions[j] = "-d "+temp;
            //System.out.println(depthOptions[j]);
        }

        for (int i = 0; i < numTrees; i++) {
            // Method 1: randomizing parameters: diversity through changing the classifier
            // diversity should be injected into the ensemble by randomising the DECISION TREE parameters
            //   -  Including attribute selection mechanisms
            //   -  Stopping conditions: Including maxDepth

            //   - take a classifier object and perform a random setOptions on the classifier object
            CourseworkTree c = new CourseworkTree();

            // get random number between zero and depthOptions.length
            int depth = rn.nextInt(depthOptions.length);
            // get random number between zero and attributeSelectionOptions.length
            int attSelectionMethod = rn.nextInt(attributeSelectionOptions.length);

            c.setOptions(Utils.splitOptions(attributeSelectionOptions[attSelectionMethod]));
            c.setOptions(Utils.splitOptions(depthOptions[depth]));
            //System.out.println("classifier "+(i+1)+" has depth: "+depthOptions[depth]);
            //System.out.println("classifier "+(i+1)+" has attribute selection method: "+attributeSelectionOptions[attSelectionMethod]);

            treeEnsemble.add(c);
        }

        // remove instances with missing class
        data = new Instances(data);
        data.deleteWithMissingClass();

        // get the size of the subspace (calculates number of attributes for new dataset
        // equal to te number of attributes wanting to remove
        double subSpaceSize = Math.round(((double)data.numAttributes()-1)*(((double)attributeSelectionSize/(double)100)));
        //System.out.println("subspacesize: "+subSpaceSize);
        int intSpaceSize = (int) subSpaceSize;
        //System.out.println(intSpaceSize);

        //attribute bag for every classifier
        attributeBags = new String[numTrees];

        int bagIndex = 0;
        // build for every classifer
        for (int j = 0; j < numTrees; j++) {
                // Method 2: selecting an attribute subset of the data; diversity through changing the data.
                //       -   attribute subsets stored internally by the Ensemble in attributeBags 2D array

                Integer[] indices = new Integer[data.numAttributes()-1];    // get array of size equal to attributes to store their index'
                int classIndex = data.classIndex();                         // get index of class
                int offset = 0;
                for(int i = 0; i < indices.length+1; i++) {                 // for all attributes and class value
                    if (i != classIndex) {                                  // fill out indicies with the index'
                        indices[offset++] = i+1;                            // for array of all attribute index'
                    }
                }

                // get random number calculator
                Random random = new Random();

                Collections.shuffle(Arrays.asList(indices), random);
                StringBuffer sb = new StringBuffer("");

                for (int i = 0; i < intSpaceSize; i++) {
                    sb.append(indices[i] + ",");
                }
                sb.append(classIndex + 1);

                // store in bags for access with classifyInstacne and distributionForInstance later
                attributeBags[bagIndex] = sb.toString();
                // System.out.println("Bag index: " + "(" + bagIndex + ")" + " Bag for classifer "+(bagIndex+1)+": " + attributeBags[bagIndex]);

                Remove rm = new Remove();
                rm.setOptions(new String[]{"-V", "-R", sb.toString()});

                rm.setInputFormat(data);  // filter capabilities are checked here
                Instances newData = Filter.useFilter(data, rm);

                // build the classifier
                treeEnsemble.get(j).buildClassifier(newData);
                bagIndex++;
        }
            // need to store attributes used in double[][] attributeBags
            // store the array of attributes in the appropriate index of te attributeBags array
            //attributeBags[bagIndex] = attributeBag;

            // remove the attributes(data with index's removed and concatenated)

            // create a new dataset only including class value and indexed in the current attribute bag

            // build the current classifier on that dataset and do the same for the next classifier with new bag index
            // build model on the above
    }

    // Count how many classifiers predict each class value, then return the class that receives the largest no.
    public double classifyInstance(Instance inst) throws Exception{
        // count for each class
        int[] votes=new int[inst.numClasses()];

        // 1. ask each base classifier (ensemble members) for prediction and count predictions for each class
        for(Classifier c:treeEnsemble){
            //cast to int as classifyInstance returns double
            votes[(int)c.classifyInstance(inst)]++;
        }

        // 2. return the one with the highest count (most votes)
        // Find index of the highest number of votes as this will be the class value we predict
        int argMax=0;
        // assume argMax is zero so just need to check if the next index' are higher
        for(int i=1;i<votes.length;i++) {
            if (votes[i] > votes[argMax]) {
                argMax = i;
            }
        }
        return argMax;
    }

    // return proportion of votes for each class or averages probabilities depending on the value of
    // averageDistributions
    public double[] distributionForInstance(Instance inst) throws Exception {
        // average the probabilities
        if (averageDistributions) {
            double[] probs= new double[inst.numClasses()];
            // for classifier c gets the proportion of 'probability estimates' for each class value
            for(Classifier c:treeEnsemble){
                //cast to int as classifyInstance returns double
                double[] d = c.distributionForInstance(inst);
                // sum respective class value probabilities
                for (int i=0; i<d.length; i++) {
                    probs[i]+=d[i];
                }
            }
            double sum = 0;
            for(int i = 0; i<probs.length; i++){
                // get total probabilites summed for all class values
                sum+=probs[i];
            }
            for(int i = 0;i<probs.length; i++){
                // divide totalized class probabilities for each class by the total probabilites for all classes to get
                // the distribution of averaged probabilities for each class
                probs[i]/=sum;
            }
            return probs;
        }
        else {
            // proportion of 'votes' for each class
            double[] votes=new double[inst.numClasses()];

            // 1. ask each base classifier (ensemble members) for prediction and count predictions for each class
            for(Classifier c:treeEnsemble){
                //cast to int as classifyInstance returns double
                votes[(int)c.classifyInstance(inst)]++;
            }

            // 2. return the proportion of votes for each label
            // Find total votes then divide individual votes by the total
            int total = 0;
            for(int i=0;i<votes.length;i++) {
                total += votes[i];
            }
            for(int i=0;i<votes.length;i++) {
                votes[i] /= total;
            }
            return votes;
        }
    }


    // Load optdigits and ChinaTown, print out the test accuracy, also print out the probability estimates for
    // first five test cases
    public static void main(String[] argv) throws Exception {
        Instances nominal = DatasetLoading.loadData("src\\main\\java\\ml_6002b_coursework\\test_data\\optdigits.arff");
        Instances numeric = DatasetLoading.loadData("src\\main\\java\\ml_6002b_coursework\\test_data\\Chinetown.arff");
        // System.out.println(numeric);

        // Discretize Numeric Values in dataset
        AttributeSplitMeasure am = new IGAttributeSplitMeasure();
        // Leave out class attribute
        for (int k = 0; k < numeric.numAttributes()-1; k++) {
            numeric = am.splitDataOnNumeric(numeric, numeric.attribute(k));
        }
        // System.out.println(numeric);

        // convert attribute labels using weka's NumericToNominal filter
        NumericToNominal convert= new NumericToNominal();
        String[] options= new String[2];
        options[0]="-R";
        options[1]="1-24";  //range of variables to make numeric
        convert.setOptions(options);
        convert.setInputFormat(numeric);
        numeric = Filter.useFilter(numeric, convert);
        //System.out.println(numeric);

        TreeEnsemble treeEnsembleNom = new TreeEnsemble();
        treeEnsembleNom.buildClassifier(nominal);
        TreeEnsemble treeEnsembleNum = new TreeEnsemble();
        treeEnsembleNum.buildClassifier(numeric);

        // test nominal
        int countNom = 0;
        int totalNom = 0;
        int tests = 1;
        for(int k = 0; k < treeEnsembleNom.numTrees; k++) {
            // need to store which attributes are used with which classifier (attributeBags[][]) in order to
            // recreate the attribute selections
            // get random number calculator
            Random random = new Random();
            StringBuffer sb = new StringBuffer("");
            sb.append(treeEnsembleNom.attributeBags[k]);

            Remove rm = new Remove();
            rm.setOptions(new String[]{"-V", "-R", sb.toString()});

            rm.setInputFormat(nominal);  // filter capabilities are checked here
            Instances newData = Filter.useFilter(nominal, rm);

            //System.out.println("\n"+sb.toString());
            //System.out.println("\n\n"+newData);

            int classifierAccuracy = 0;
            for (Instance inst : newData) {
                int pred = (int) treeEnsembleNom.treeEnsemble.get(k).classifyInstance(inst);
                int actual = (int) inst.classValue();
                if(pred==actual){
                    countNom++;
                    classifierAccuracy++;
                }
                totalNom++;

                // print out nominal probability estimates for first 5 tests
                treeEnsembleNom.setAverageDistributions(true);
                double[] dist1 = treeEnsembleNom.treeEnsemble.get(k).distributionForInstance(inst);
                //System.out.println("distribution length: "+dist1.length);
                treeEnsembleNom.setAverageDistributions(false);
                double[] dist2 = treeEnsembleNom.treeEnsemble.get(k).distributionForInstance(inst);
                if (tests < 6) {
                    System.out.println("\n\nNominal Test "+tests+" using Classifier "+(k+1)+" on Instance " + tests + " of optdigits with attribute subset "+(k+1) +":");
                    for (int j = 0; j < dist1.length; j++) {
                        System.out.println("Averaged distributions for class (" + j + ") = " + dist1[j]);
                    }
                    for (int j = 0; j < dist2.length; j++) {
                        System.out.println("Proportion of votes for class (" + j + ") = " + dist2[j]);
                    }
                    System.out.println("\n");
                }
                tests++;
            }
            System.out.println("Nominal_Classifier_"+(k+1)+" test accuracy = " + (double) classifierAccuracy / (double) newData.numInstances());
        }


        // test numeric
        int countNum = 0;
        int totalNum = 0;
        tests = 1;
        for(int t = 0; t < treeEnsembleNum.numTrees; t++) {
            // need to store which attributes are used with which classifier (attributeBags[][]) in order to
            // recreate the attribute selections
            // get random number calculator
            Random random = new Random();
            StringBuffer sb = new StringBuffer("");
            sb.append(treeEnsembleNum.attributeBags[t]);

            Remove rm = new Remove();
            rm.setOptions(new String[]{"-V", "-R", sb.toString()});

            rm.setInputFormat(numeric);  // filter capabilities are checked here
            Instances newData = Filter.useFilter(numeric, rm);

            //System.out.println("\n"+sb.toString());
            //System.out.println("\n\n"+newData);

            int classifierAccuracy = 0;
            for (Instance inst : newData) {
                int pred = (int) treeEnsembleNum.treeEnsemble.get(t).classifyInstance(inst);
                int actual = (int) inst.classValue();
                if(pred==actual){
                    countNum++;
                    classifierAccuracy++;
                }
                totalNum++;

                // print out nominal probability estimates for first 5 tests
                treeEnsembleNum.setAverageDistributions(true);
                double[] dist3 = treeEnsembleNum.treeEnsemble.get(t).distributionForInstance(inst);
                //System.out.println("distribution length: "+dist1.length);
                treeEnsembleNum.setAverageDistributions(false);
                double[] dist4 = treeEnsembleNum.treeEnsemble.get(t).distributionForInstance(inst);
                if (tests < 6) {
                    System.out.println("\n\nNumeric Test "+tests+" using Classifier "+(t+1)+" on Instance " + tests + " of chinetown with attribute subset "+(t+1) +":");
                    for (int j = 0; j < dist3.length; j++) {
                        System.out.println("Averaged distributions for class (" + j + ") = " + dist3[j]);
                    }
                    for (int j = 0; j < dist4.length; j++) {
                        System.out.println("Proportion of votes for class (" + j + ") = " + dist4[j]);
                    }
                    System.out.println("\n");
                }
                tests++;
            }
            System.out.println("Numeric_Classifier_"+(t+1)+" test accuracy = " + (double) classifierAccuracy / (double) newData.numInstances());
        }

        double numAccuracy  =  (double) countNum / (double) totalNum;
        double nomAccuracy  =  (double) countNom / (double) totalNom;

        System.out.println("\nNumeric_Ensemble test accuracy: "+ numAccuracy);
        System.out.println("Nominal_Ensemble test accuracy: "+ nomAccuracy);
    }
}
