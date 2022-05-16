package ml_6002b_coursework;

import experiments.data.DatasetLoading;
import utilities.InstanceTools;
import weka.classifiers.AbstractClassifier;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.SimpleFilter;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.util.Arrays;
import java.util.Random;

/**
 * A basic decision tree classifier for use in machine learning coursework (6002B).
 */
public class CourseworkTree extends AbstractClassifier {

    /** Measure to use when selecting an attribute to split the data with. */
    private AttributeSplitMeasure attSplitMeasure = new IGAttributeSplitMeasure();

    /** Maxiumum depth for the tree. */
    private int maxDepth = Integer.MAX_VALUE;

    /** The root node of the tree. */
    private TreeNode root;

    /** Stores the String of splits made upon building **/
    private String splitCode = "";

    /**
     * Sets the attribute split measure for the classifier.
     *
     * @param attSplitMeasure the split measure
     */
    public void setAttSplitMeasure(AttributeSplitMeasure attSplitMeasure) {
        this.attSplitMeasure = attSplitMeasure;
    }

    /**
     * Sets the max depth for the classifier.
     *
     * @param maxDepth the max depth
     */
    public void setMaxDepth(int maxDepth){
        this.maxDepth = maxDepth;
    }

    /**
     * Returns default capabilities of the classifier.
     *
     * @return the capabilities of this classifier
     */
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);

        //instances
        result.setMinimumNumberInstances(2);

        return result;
    }

    /**
     * Builds a decision tree classifier.
     *
     * @param data the training data
     */
    @Override
    public void buildClassifier(Instances data) throws Exception {
        // 0.1. Stopping criteria
        if (data.classIndex() != data.numAttributes() - 1) {
            throw new Exception("Class attribute must be the last index.");
        }

        // 0.2. Initialize Tree-node and build tree
        root = new TreeNode();
        root.buildTree(data, 0);
    }

    /**
     * Classifies a given test instance using the decision tree.
     *
     * @param instance the instance to be classified
     * @return the classification
     */
    @Override
    public double classifyInstance(Instance instance) {
        double[] probs = distributionForInstance(instance);

        int maxClass = 0;
        for (int n = 1; n < probs.length; n++) {
            if (probs[n] > probs[maxClass]) {
                maxClass = n;
            }
        }

        return maxClass;
    }

    /**
     * Computes class distribution for instance using the decision tree.
     *
     * @param instance the instance for which distribution is to be computed
     * @return the class distribution for the given instance
     */
    @Override
    public double[] distributionForInstance(Instance instance) {
        return root.distributionForInstance(instance);
    }

    /**
     * Class representing a single node in the tree.
     */
    private class TreeNode {

        /** Attribute used for splitting, if null the node is a leaf. */
        Attribute bestSplit = null;

        /** Best gain from the splitting measure if the node is not a leaf. */
        double bestGain = 0;

        /** Depth of the node in the tree. */
        int depth;

        /** The node's children if it is not a leaf. */
        TreeNode[] children;

        /** The class distribution if the node is a leaf. */
        double[] leafDistribution;

        /**
         * Recursive function for building the tree.
         * Builds a single tree node, finding the best attribute to split on using a splitting measure.
         * Splits the best attribute into multiple child tree node's if they can be made, else creates a leaf node.
         *
         * @param data Instances to build the tree node with
         * @param depth the depth of the node in the tree
         */
        void buildTree(Instances data, int depth) throws Exception {
            this.depth = depth;

            // Choosing an attribute
            // 1. Loop through each attribute, finding the best one.
            for (int i = 0; i < data.numAttributes() - 1; i++) {
                double gain = attSplitMeasure.computeAttributeQuality(data, data.attribute(i));

                if (gain > bestGain) {
                    bestSplit = data.attribute(i);
                    bestGain = gain;
                }
            }

            // 2. Split data by attribute
            // If we found an attribute to split on, create child nodes.
            if (bestSplit != null) {
                Instances[] split = attSplitMeasure.splitData(data, bestSplit);
                children = new TreeNode[split.length];

                // store split attribute name in split code
                if (splitCode == "") {
                    splitCode += bestSplit.name();
                }
                else {
                    splitCode += "-"+bestSplit.name();
                }
                //System.out.println("SPLIT ON: "+splitCode);

                // 3. Recursively call for each split:
                // Create a child for each value in the selected attribute, and determine whether it is a leaf or not.
                for (int i = 0; i < children.length; i++){
                    children[i] = new TreeNode();

                    boolean leaf = split[i].numDistinctValues(data.classIndex()) == 1 || depth + 1 == maxDepth;

                    if (split[i].isEmpty()) {
                        children[i].buildLeaf(data, depth + 1);
                    } else if (leaf) {
                        children[i].buildLeaf(split[i], depth + 1);
                    } else {
                        children[i].buildTree(split[i], depth + 1);
                    }
                }
                // Else turn this node into a leaf node.
            } else {
                leafDistribution = classDistribution(data);
            }
        }

        /**
         * Builds a leaf node for the tree, setting the depth and recording the class distribution of the remaining
         * instances.
         *
         * @param data remaining Instances to build the leafs class distribution
         * @param depth the depth of the node in the tree
         */
        void buildLeaf(Instances data, int depth) {
            this.depth = depth;
            leafDistribution = classDistribution(data);
        }

        /**
         * Recursive function traversing node's of the tree until a leaf is found. Returns the leafs class distribution.
         *
         * @return the class distribution of the first leaf node
         */
        double[] distributionForInstance(Instance inst) {
            // If the node is a leaf return the distribution, else select the next node based on the best attributes
            // value.
            if (bestSplit == null) {
                return leafDistribution;
            } else {
                return children[(int) inst.value(bestSplit)].distributionForInstance(inst);
            }
        }

        /**
         * Returns the normalised version of the input array with values summing to 1.
         *
         * @return the class distribution as an array
         */
        double[] classDistribution(Instances data) {
            double[] distribution = new double[data.numClasses()];
            for (Instance inst : data) {
                distribution[(int) inst.classValue()]++;
            }

            double sum = 0;
            for (double d : distribution){
                sum += d;
            }

            if (sum != 0){
                for (int i = 0; i < distribution.length; i++) {
                    distribution[i] = distribution[i] / sum;
                }
            }

            return distribution;
        }

        /**
         * Summarises the tree node into a String.
         *
         * @return the summarised node as a String
         */
        @Override
        public String toString() {
            String str;
            if (bestSplit == null){
                str = "Leaf," + Arrays.toString(leafDistribution) + "," + depth;
            } else {
                str = bestSplit.name() + "," + bestGain + "," + depth;
            }
            return str;
        }
    }

    /**
     * Main method.
     *
     * @param args the options for the classifier main
     */
    public static void main(String[] args) throws Exception {
        Instances nominal = DatasetLoading.loadData("src\\main\\java\\ml_6002b_coursework\\test_data\\optdigits.arff");
        Instances numeric = DatasetLoading.loadData("src\\main\\java\\ml_6002b_coursework\\test_data\\Chinetown.arff");
        //System.out.println(numeric);

        // Discretize Numeric Values in dataset
        AttributeSplitMeasure am = new IGAttributeSplitMeasure();
        for (int k = 0; k < numeric.numAttributes(); k++) {
            numeric = am.splitDataOnNumeric(numeric, numeric.attribute(k));
        }
        //System.out.println(numeric);

        // convert attribute labels using weka's NumericToNominal filter
        NumericToNominal convert= new NumericToNominal();
        String[] options= new String[2];
        options[0]="-R";
        options[1]="1-24";  //range of variables to make numeric
        convert.setOptions(options);
        convert.setInputFormat(numeric);
        numeric = Filter.useFilter(numeric, convert);
        //System.out.println(numeric);

        // make a random train/test split for the two datasets using weka's resampleInstances() utility
        Instances[] splitNumeric = InstanceTools.resampleInstances(numeric, 0, 0.1);
        Instances[] splitNominal = InstanceTools.resampleInstances(nominal, 0, 0.1);

        // initialize info gain trees
        CourseworkTree treeIGOptdigits = new CourseworkTree();
        treeIGOptdigits.setOptions(Utils.splitOptions("-gain 1 -depth 3"));
        CourseworkTree treeIGChinetown = new CourseworkTree();
        treeIGChinetown.setOptions(Utils.splitOptions("-gain 1 -depth 3"));
        // initialize info gain ratio trees
        CourseworkTree treeIGROptdigits = new CourseworkTree();
        treeIGROptdigits.setOptions(Utils.splitOptions("-ratio 1 -depth 3"));
        CourseworkTree treeIGRChinetown = new CourseworkTree();
        treeIGRChinetown.setOptions(Utils.splitOptions("-ratio 1 -depth 3"));
        // initialize chi squared statistic trees
        CourseworkTree treeChiOptdigits = new CourseworkTree();
        treeChiOptdigits.setOptions(Utils.splitOptions("-chi 1 -depth 3"));
        CourseworkTree treeChiChinetown = new CourseworkTree();
        treeChiChinetown.setOptions(Utils.splitOptions("-chi 1 -depth 3"));
        // initialize gini index trees
        CourseworkTree treeGiniOptdigits = new CourseworkTree();
        treeGiniOptdigits.setOptions(Utils.splitOptions("-gini 1 -depth 3"));
        CourseworkTree treeGiniChinetown = new CourseworkTree();
        treeGiniChinetown.setOptions(Utils.splitOptions("-gini 1 -depth 3"));

        // get random split nominal
//        Random r1 = new Random();
//        int lowBoundNom = 1;
//        int highBoundNom = 64;
//        int randomNominalAttribute = r1.nextInt(highBoundNom-lowBoundNom) + lowBoundNom;
        // get random split numeric
//        Random r2 = new Random();
//        int lowBoundNum = 0;
//        int highBoundNum = 23;
//        int randomNumericAttribute = r2.nextInt(highBoundNum-lowBoundNum) + lowBoundNum;

        // chi tree optdigits and chinetown random splits
//        Instances[] optdigitsChiNominal = treeChiOptdigits.attSplitMeasure.splitData(nominal, nominal.attribute(randomNominalAttribute));
//        Instances[] chinetownChiNumeric = treeChiChinetown.attSplitMeasure.splitData(numeric, numeric.attribute(randomNumericAttribute));
        // info gain optdigits and chinetown random splits
//        Instances[] optdigitsInfoGainNominal = treeIGOptdigits.attSplitMeasure.splitData(nominal, nominal.attribute(randomNominalAttribute));
//        Instances[] chinetownInfoGainNumeric = treeIGChinetown.attSplitMeasure.splitData(numeric, numeric.attribute(randomNumericAttribute));
        // info gain ratio optdigits and chinetown random splits
//        Instances[] optdigitsInfoGainRatioNominal = treeIGROptdigits.attSplitMeasure.splitData(nominal, nominal.attribute(randomNominalAttribute));
//        Instances[] chinetownInfoGainRatioNumeric = treeIGRChinetown.attSplitMeasure.splitData(numeric, numeric.attribute(randomNumericAttribute));
        // gini index optdigits and chinetown random splits
//        Instances[] optdigitsGiniNominal = treeGiniOptdigits.attSplitMeasure.splitData(nominal, nominal.attribute(randomNominalAttribute));
//        Instances[] chinetownGiniNumeric = treeGiniChinetown.attSplitMeasure.splitData(numeric, numeric.attribute(randomNumericAttribute));


        // build classifiers using chi, info and gini:
        // {build classifiers on the first array}
        // chi built DT's
        treeChiOptdigits.buildClassifier(splitNominal[0]);
        treeChiChinetown.buildClassifier(splitNumeric[0]);
        //System.out.println(treeChiOptdigits.root.toString());
        //System.out.println(treeChiChinetown.root.toString());

        // info gain built DT's
        treeIGOptdigits.buildClassifier(splitNominal[0]);
        treeIGChinetown.buildClassifier(splitNumeric[0]);
        //System.out.println(treeIGOptdigits.root.toString());
        //System.out.println(treeIGChinetown.root.toString());

        // info gain ratio built DT's
        treeIGROptdigits.buildClassifier(splitNominal[0]);
        treeIGRChinetown.buildClassifier(splitNumeric[0]);
        //System.out.println(treeIGROptdigits.root.toString());
        //System.out.println(treeIGRChinetown.root.toString());

        // gini built DT's
        treeGiniOptdigits.buildClassifier(splitNominal[0]);
        treeGiniChinetown.buildClassifier(splitNumeric[0]);
        //System.out.println(treeGiniOptdigits.root.toString());
        //System.out.println(treeGiniChinetown.root.toString());


        // Output the test accuracy of each optdigits tree built
        // {test classifiers on the second array}
        // In the form:
        // DT using measure <insert> on optdigits problem has test accuracy = <insert>
        int chiOptdigitsCount = 0;
        int infoGainOptdigitsCount = 0;
        int infoGainRatioOptdigitsCount = 0;
        int giniOptdigitsCount = 0;
        for(Instance i:splitNominal[1]){
            //models
            double predictedChiOptdigits = treeChiOptdigits.classifyInstance(i);
            double predictedInfoGainOptdigits = treeIGOptdigits.classifyInstance(i);
            double predictedInfoGainRatioOptdigits = treeIGROptdigits.classifyInstance(i);
            double predictedGiniOptdigits = treeGiniOptdigits.classifyInstance(i);

            //actual result
            double actual = i.classValue();
            //System.out.println(i);
            //System.out.println(actual);

            // check predictions
            if(predictedChiOptdigits==actual)
                chiOptdigitsCount++;
            if(predictedInfoGainOptdigits==actual)
                infoGainOptdigitsCount++;
            if(predictedInfoGainRatioOptdigits==actual)
                infoGainRatioOptdigitsCount++;
            if(predictedGiniOptdigits==actual)
                giniOptdigitsCount++;
        }
        System.out.println("DT using measure Chi Squared on optdigits problem has test accuracy: "+ chiOptdigitsCount/(double)splitNominal[1].numInstances());
        System.out.println("DT using measure Information Gain on optdigits problem has test accuracy: "+ infoGainOptdigitsCount/(double)splitNominal[1].numInstances());
        System.out.println("DT using measure Information Gain Ratio on optdigits problem has test accuracy: "+ infoGainRatioOptdigitsCount/(double)splitNominal[1].numInstances());
        System.out.println("DT using measure Gini Index on optdigits problem has test accuracy: "+ giniOptdigitsCount/(double)splitNominal[1].numInstances());

        // Output the test accuracy of each chinetown tree built
        // {test classifiers on the second array}
        // In the form:
        // DT using measure <insert> on chinetown problem has test accuracy = <insert>
        int chiChinetownCount = 0;
        int infoGainChinetownCount = 0;
        int infoGainRatioChinetownCount = 0;
        int giniChinetownCount = 0;
        for(Instance i:splitNumeric[1]){
            //models
            double predictedChiChinetown = treeChiChinetown.classifyInstance(i);
            double predictedInfoGainChinetown = treeIGChinetown.classifyInstance(i);
            double predictedInfoGainRatioChinetown = treeIGRChinetown.classifyInstance(i);
            double predictedGiniChinetown = treeGiniChinetown.classifyInstance(i);

            //actual result (converted from {1 ,2} to {0, 1})
            double actual = i.classValue();
//            System.out.println(i);
//            System.out.println(actual);
//            System.out.println("   Chi Predicted = "+predictedChiChinetown);
//            System.out.println("   InfoGain Predicted = "+predictedInfoGainChinetown);
//            System.out.println("   InfoGainRatio Predicted = "+predictedInfoGainRatioChinetown);
//            System.out.println("   Gini Predicted = "+predictedGiniChinetown);

            // check predictions
            if(predictedChiChinetown==actual)
                chiChinetownCount++;
            if(predictedInfoGainChinetown==actual)
                infoGainChinetownCount++;
            if(predictedInfoGainRatioChinetown==actual)
                infoGainRatioChinetownCount++;
            if(predictedGiniChinetown==actual)
                giniChinetownCount++;
        }
//        System.out.println(chiChinetownCount);
//        System.out.println(infoGainChinetownCount);
//        System.out.println(infoGainRatioChinetownCount);
//        System.out.println(giniChinetownCount);
        System.out.println("\n\nDT using measure Chi Squared on chinetown problem has test accuracy: "+ chiChinetownCount/(double)splitNumeric[1].numInstances());
        System.out.println("DT using measure Information Gain on chinetown problem has test accuracy: "+ infoGainChinetownCount/(double)splitNumeric[1].numInstances());
        System.out.println("DT using measure Information Gain Ratio on chinetown problem has test accuracy: "+ infoGainRatioChinetownCount/(double)splitNumeric[1].numInstances());
        System.out.println("DT using measure Gini Index on chinetown problem has test accuracy: "+ giniChinetownCount/(double)splitNumeric[1].numInstances());
    }

    public void setOptions(String[] options) throws Exception {
        String tmpStr;

        tmpStr = Utils.getOption("depth", options);
        if (tmpStr.length() != 0) {
            setMaxDepth(Integer.parseInt(tmpStr));
        } else {
            setMaxDepth(1);
        }

        tmpStr = Utils.getOption("chi", options);
        if (tmpStr.length() != 0) {
            ChiSquaredAttributeSplitMeasure chi = new ChiSquaredAttributeSplitMeasure();
            setAttSplitMeasure(chi);
        }

        tmpStr = Utils.getOption("gain", options);
        if (tmpStr.length() != 0) {
            IGAttributeSplitMeasure ig = new IGAttributeSplitMeasure();
            ig.useGain = true;
            setAttSplitMeasure(ig);
        }

        tmpStr = Utils.getOption("gini", options);
        if (tmpStr.length() != 0) {
            GiniAttributeSplitMeasure gini = new GiniAttributeSplitMeasure();
            setAttSplitMeasure(gini);
        }

        tmpStr = Utils.getOption("ratio", options);
        if (tmpStr.length() != 0) {
            IGAttributeSplitMeasure ig = new IGAttributeSplitMeasure();
            ig.useGain = false;
            setAttSplitMeasure(ig);
        }

        super.setOptions(options);
        Utils.checkForRemainingOptions(options);
    }

}
