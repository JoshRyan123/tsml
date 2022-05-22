package ml_6002b_coursework;

import experiments.data.DatasetLoading;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.text.DecimalFormat;

/**
 * Empty class for Part 2.1 of the coursework.
 */
public class AttributeMeasures {
    // Should follow formula shown in lecture 2, and comment code to indicate edge cases

    // returns WEIGHTED info gain (entropy) for contingency table
    static double measureInformationGain(int[][] table) {
        double Entropy;
        double Bag[] = new double [table.length];
        double Class[] = new double [table[0].length];
        double total = 0;
        double oldEntropy = 0;
        double splitEntropy = 0;
        int i, j;

        // iterate through rows (attribute values) in contingency table
        for (i = 0; i < table.length; i++) {
            // iterate through columns (class values) of contingency table
            for (j = 0; j < table[i].length; j++) {
                Bag[i] += table[i][j];
                Class[j] += table[i][j];
                total += table[i][j];
                //System.out.println("Bag i: "+Bag[i]);
                //System.out.println("Class j: "+Class[j]);
            }
        }
//        for (i = 0; i < Class.length; i++) {
//            System.out.println("class counts: "+ Class[i]);
//        }

        double oldValue = 0;
        int p;
        for (p=0;p<Class.length;p++) {
            oldValue = oldValue + logFunc(Class[p]);
            //System.out.println("log function old value"+logFunc(Class[p]));
        }
        // calculate old entropy
        oldEntropy = oldEntropy+(logFunc(total)-oldValue);
        // System.out.println("old entropy: "+oldEntropy);

        double newValue = 0;
        int k,o;
        for (k=0;k<Bag.length;k++){
            for (o=0;o<Class.length;o++)
                newValue = newValue+logFunc(table[k][o]);
                // deal with NaN calculations in entropy
                if (Double.isNaN(newValue))
                    newValue = 0;
            //System.out.println("log function new value"+logFunc(Bag[k]));
            newValue = newValue-logFunc(Bag[k]);
        }
        // calculate split entropy
        splitEntropy -= newValue;

        // calculate total entropy
        Entropy = total/(oldEntropy-splitEntropy);

        // splits with no gain are useless
        if (Utils.eq(Entropy,0))
            return Double.MAX_VALUE;

        return Entropy;
    }
    public static final double logFunc(double num) {
        double log2 = Math.log(2);
        if (num < 1e-6)
            return 0;
        else
            return num*Math.log(num)/log2;
    }


    // returns info gain ratio for contingency table
    static double measureInformationGainRatio(int[][] table) throws Exception {
        // entropy is 1 as 50/50 split (base entropy)
        int k, p;
        double total = 0;
        double [] classCounts = new double[table[0].length];
        double parentEnt = 0;
        double childEnt = 0;

        // iterate through rows (attribute values) in contingency table
        for (k = 0; k < table.length; k++) {
            // iterate through columns (class values) of contingency table
            for (p = 0; p < table[k].length; p++) {
                classCounts[p] += table[k][p];
                total += table[k][p];
            }
        }
//        for (k = 0; k < classCounts.length; k++) {
//            System.out.println("class counts: "+ classCounts[k]);
//        }

        double baseEntropy = 0;
        for (int j = 0; j < classCounts.length; j++) {
            baseEntropy = baseEntropy + logFunc(classCounts[j]);
        }
        parentEnt = parentEnt+(logFunc(total)-baseEntropy);
        parentEnt /= total;
        // System.out.println("parentEnt:"+parentEnt);

        // set info gain equal to starting entropy
        double splitDist;
        double splitInfo = 0;
        int i, j;
        for (i = 0; i < table.length; i++) {
            for (j = 0; j < classCounts.length; j++) {
                // take individual split count and divide by weighting for whole split
                splitDist = (double) (table[i][j]) /
                        ((Utils.sum(table[i])));
                //System.out.println(Utils.sum(table[i]));
                //System.out.println(total);
                // deal with NaN calculations in entropy
                if (Double.isNaN(splitDist))
                    splitDist = 0;
                //calculate entropy and simtiply by the total split distribution
                childEnt -=  (Utils.sum(table[i])/total)*(logFunc(splitDist));

//                System.out.println("split distribution:"+splitDist);
//                System.out.println("log distribution: "+logFunc(splitDist));
//                System.out.println("resultsing childENt: "+childEnt);
            }
            // CALCULATE THE SPLIT_INFO(data, att) to normalize the Gain
            // SPLIT_INFO = (totalSplitWeight/totalParentWeight)*log(totalSplitWeight/totalParentWeight)
            // i.e 6/10 * log(6/10)
            //          +
            //     4/10 * log(4/10)
            //          =
            //       splitInfo
            // then do:
            //       GainRatio = Entropy/splitInfo
            splitInfo -= logFunc(Utils.sum(table[i])/total);
            //System.out.println("splitInfo = "+ splitInfo);
        }
        //System.out.println("childEnt: "+childEnt);
        double infoGain = parentEnt-childEnt;
        //System.out.println("parent ent: "+parentEnt);
        //System.out.println("parent ent minus child ent: "+infoGain);

        infoGain = infoGain/splitInfo;

        return infoGain;
    }

    // returns gini measure for contingency table
    static double measureGini(int[][] table) {
        double[][] t = new double[table.length][table[0].length];

        // entropy is 1 as 50/50 split (base entropy)
        int k, p;
        double [] classCounts = new double[table[0].length];

        for (k = 0; k < table.length; k++) {
            // iterate through columns (class values) of contingency table
            for (p = 0; p < table[k].length; p++) {
                classCounts[p] += table[k][p];
                t[k][p] = table[k][p];
                // System.out.println(table[k][p]);
            }
        }

        double totalWeight = Utils.sum(classCounts);
        if (totalWeight==0) return 0;

        double leftWeight = Utils.sum(t[0]);
        double rightWeight = Utils.sum(t[1]);
        //System.out.println("left wight:"+leftWeight);
        //System.out.println("right wight:"+rightWeight);

        double parentVal = 0;
        for (int i=0; i<classCounts.length; i++) {
            parentVal += (classCounts[i]/totalWeight)*(classCounts[i]/totalWeight);
        }
        // parent impurity measure
        parentVal = 1-parentVal;
        // System.out.println("parent gini index: "+parentVal);

        double leftVal = 0;
            for (int i = 0; i < t[0].length; i++) {
                leftVal += (t[0][i] / leftWeight) * (t[0][i] / leftWeight);
                //System.out.println("childDistributions[0][i] / leftWeight = " +leftVal);
                // deal with NaN calculations in left gini value
                if (Double.isNaN(leftVal))
                    leftVal = 0;
            }
            leftVal = 1-leftVal;


        // System.out.println("left gini index: "+leftVal);

        double rightVal = 0;
        for (int i=0; i<t[1].length; i++) {
            rightVal += (t[1][i]/rightWeight)*(t[1][i]/rightWeight);
            //System.out.println("childDistributions[1][i] / rightWeight = " +rightVal);
            // deal with NaN calculations in right gini value
            if (Double.isNaN(rightVal))
                rightVal = 1;
        }
        rightVal = 1-rightVal;

        // System.out.println("right gini index: "+rightVal);

        //System.out.println("final parent val:"+childDistributions[1][1]+" "+childDistributions[1][0]);
        //System.out.println("final left val:"+leftVal);
        //System.out.println("final right val:"+rightVal);

        // 1 - 0.5 - 0.72 - 1
        return parentVal - leftWeight/totalWeight*leftVal - rightWeight/totalWeight*rightVal;
    }

    // returns chi statistic for contingency table
    static double measureChiSquared(int[][] table) {
        double chiSquaredStatistic;

        double[][] t = new double[table.length][table[0].length];

        // entropy is 1 as 50/50 split (base entropy)
        int k, p;
        double [] classCounts = new double[table[0].length];
        for (k = 0; k < table.length; k++) {
            // iterate through columns (class values) of contingency table
            for (p = 0; p < table[k].length; p++) {
                classCounts[p] += table[k][p];
                t[k][p] = table[k][p];
                // System.out.println(table[k][p]);
            }
        }
        // System.out.println("parent distribution: "+ parentDistribution[0]+" "+parentDistribution[1]);

        double totalWeight = Utils.sum(classCounts);
        double leftWeight = Utils.sum(t[0]);
        double rightWeight = Utils.sum(t[1]);
        if (totalWeight==0) return 0;

        // expected probabilies left: expectedProbabilities[0]
        // expected probabilies right: expectedProbabilities[1]
        // get expected probabilities from parent distribution and total parent weight
        double[] expectedProbabilities = new double[classCounts.length];
        // expected left is total left over each left varaible
        double[] expectedLeftSplit = new double[classCounts.length];
        double[] actualLeftSplit = new double[classCounts.length];
        double[] expectedRightSplit = new double[classCounts.length];
        double[] actualRightSplit = new double[classCounts.length];
        int i;
        for (i = 0; i < expectedProbabilities.length; i++) {
            expectedProbabilities[i] = classCounts[i]/totalWeight;
        }

        // expected equal to total left multiplied by expected distribution
        for (i = 0; i < expectedLeftSplit.length; i++) {
            expectedLeftSplit[i] = leftWeight*expectedProbabilities[i];
        }
        for (i = 0; i < actualLeftSplit.length; i++) {
            actualLeftSplit[i] = table[0][i];
        }

        // expected equal to total right multiplied by expected distribution
        for (i = 0; i < expectedRightSplit.length; i++) {
            expectedRightSplit[i] = rightWeight*expectedProbabilities[i];
        }
        for (i = 0; i < actualRightSplit.length; i++) {
            actualRightSplit[i] = table[1][i];
        }

        // calculate statistics
        double[] statisticsRight = new double[classCounts.length*2];
        for (i = 0; i < actualRightSplit.length; i++) {
            statisticsRight[i] = (Math.pow(actualRightSplit[i], 2)-Math.pow(expectedRightSplit[i], 2))/expectedRightSplit[i];
            // deal with NaN calculations in right chi squared statistics
            if (Double.isNaN(statisticsRight[i]))
                statisticsRight[i] = 0;

        }
        double[] statisticsLeft = new double[classCounts.length*2];
        for (i = 0; i < actualRightSplit.length; i++) {
            statisticsLeft[i] = (Math.pow(actualLeftSplit[i], 2)-Math.pow(expectedLeftSplit[i], 2))/expectedLeftSplit[i];
            // deal with NaN calculations in left chi squared statistics
            if (Double.isNaN(statisticsLeft[i]))
                statisticsLeft[i] = 0;
        }
        chiSquaredStatistic = Utils.sum(statisticsLeft)+Utils.sum(statisticsRight);

//        System.out.println("\nExpected probabilities: "+"{"+expectedProbabilities[1]+" , "+expectedProbabilities[0]+"}");
//        System.out.println("Parent distribution right:"+parentDistribution[1] +"\nparent distribution left:"+ parentDistribution[0]);
//        System.out.println("Left Islay: "+childDistributions[0][0] +", left Speyside: "+ childDistributions[0][1]+", right Islay: "+ childDistributions[1][0]+", right Speyside: "+ childDistributions[1][1]+"\n");
//        System.out.println("Total weight: "+totalWeight +", left weight: "+ leftWeight+", right weight: "+ rightWeight+"\n");
//
//        System.out.println("Expected left split = {"+expectedLeftSplit[1]+" , "+expectedLeftSplit[0]+"}");
//        System.out.println("Actual left split = {"+actualLeftSplit[1]+" , "+actualLeftSplit[0]+"}");
//        System.out.println("Expected right split = {"+expectedRightSplit[1]+" , "+expectedRightSplit[0]+"}");
//        System.out.println("Actual right split = {"+actualRightSplit[1]+" , "+actualRightSplit[0]+"}\n");
//
//        System.out.println("left statistic = {"+statisticsLeft[1]+" , "+statisticsLeft[0]+"}\n");
//        System.out.println("right statistic = {"+statisticsRight[1]+" , "+statisticsRight[0]+"}\n");

        return chiSquaredStatistic;
    }


    /**
     * Main method.
     *
     * @param args the options for the attribute measure main
     */
    // main method test harness should test functionality of each split measure
    // should find each measure for the attribute 'Peaty' in terms of the 'diagnosis'.
    public static void main(String[] args) throws Exception {
        // data:
        Instances data = DatasetLoading.loadData("C:\\Work\\GitHub\\tsml\\src\\main\\java\\ml_6002b_coursework\\test_data\\Whiskey.arff");

        //in order to get information gain for Peaty we have to form a table:
        // rows: attribute values (0 and 1)                0  :  17
        // column: counts of that particular class:        1  :  9
        // [num values of attribute outlook(2 total slots)][number of classes(possible values for Peaty(either 1 or 0)]
        // basically makes a 2-by-2 array as seen above
        int[][] Peaty = new int[data.attribute("Peaty").numValues()][data.numClasses()];

        // loop through data and get the attribute we want' value and its class value
        // count then amount of occurance of each possible class value (either 1 or 0) of Peaty
        // do this for the first row in Peaty and then increment for the second to fill out the 2-by-2 array
        for (Instance ins : data) {
            Peaty[(int) ins.value(0)][(int) ins.classValue()]++;
        }

        // outputs the split for Peaty: (1):5 and (0):1 on one side (when class value is 0) and
        //                              (0):4 and (1):0 on the other-side (when class value is 1)
        for (int[] x : Peaty) {
            for (int y : x)
                System.out.print(" " + y + ", ");
            System.out.print("\n");
        }

        // need to find class probabilities : proportion of each class value in overall data
        // ie : 5/6 and 1/6
        // and: 4/4 and 4/0 (pure)
        // then find the effect (info gain) that node had
        DecimalFormat df = new DecimalFormat("##.#####");
        double infoGain = measureInformationGain(Peaty);
        System.out.println(" Measure Information Gain for Peaty = " + df.format(1 / infoGain));
        double infoGainRatio = measureInformationGainRatio(Peaty);
        System.out.println(" Measure Information Gain Ratio for Peaty = " + df.format(infoGainRatio));
        double giniIndex = measureGini(Peaty);
        System.out.println(" Measure Gini Index for Peaty = " + df.format(giniIndex));
        double chiSquared = measureChiSquared(Peaty);
        System.out.println(" Measure Chi Squared Statistic for Peaty = " + df.format(chiSquared));

        String problem = "MALLAT";

        for (int i = 0; i < nominalAttributeProblems.length; i++){
            //Instances train = DatasetLoading.loadData("C:\\Work\\GitHub\\tsml\\Data\\UCI Discrete\\" + nominalAttributeProblems[i] + "\\" + nominalAttributeProblems[i] + "_TRAIN.arff");
            //System.out.println(nominalAttributeProblems[i]+" train:" + train.numInstances());
            //Instances test = DatasetLoading.loadData("C:\\Work\\GitHub\\tsml\\Data\\UCI Discrete\\" + nominalAttributeProblems[i] + "\\" + nominalAttributeProblems[i] + "_TEST.arff");
            //System.out.println(nominalAttributeProblems[i]+" test:" + test.numInstances());

            Instances total = DatasetLoading.loadData("C:\\Work\\GitHub\\tsml\\Data\\MALLAT\\" + "MALLAT_TRAIN.arff");
            System.out.println(problem+" total cases:" + total.numInstances());
            System.out.println(problem+" attributes:" + total.numAttributes());
            System.out.println(problem+" attributes:" + total.numClasses());

        }
    }
    public static String[] continuousAttributeProblems={
            "bank",
            "blood",
            "breast-cancer-wisc-diag",
            "breast-tissue",
            "cardiotocography-10clases",
            "ionosphere",
            "iris",
            "libras",
            "optical",
            "ozone",
            "page-blocks",
            "parkinsons",
            "planning",
            "post-operative",
            "ringnorm",
            "seeds",
            "spambase",
            "statlog-landsat",
            "steel-plates",
            "synthetic-control",
            "twonorm",
            "vertebral-column-3clases",
            "statlog-vehicle",
            "wall-following",
            "waveform-noise",
            "wine-quality-white",
            "yeast",
    };

    public static String[] allProblems={
            "car-evaluation",
            "chess-krvk",
            "chess-krvkp",
            "connect-4",
            "contraceptive-method",
            "fertility",
            "habermans-survival",
            "hayes-roth",
            "led-display",
            "lymphography",
            "molecular-promoters",
            "molecular-splice",
            "monks-1",
            "monks-2",
            "monks-3",
            "nursery",
            "optdigits",
            "pendigits",
            "semeion",
            "spect-heart",
            "tic-tac-toe",
            "zoo",
    };

    public static String[] nominalAttributeProblems={
            "balance-scale",
            "chess-krvk",
            "chess-krvkp",
            "connect-4",
            "contraceptive-method",
            "fertility",
            "habermans-survival",
            "hayes-roth",
            "led-display",
            "lymphography",
            "molecular-promoters",
            "molecular-splice",
            "monks-1",
            "monks-2",
            "monks-3",
            "nursery",
            "optdigits",
            "pendigits",
            "semeion",
            "spect-heart",
            "tic-tac-toe",
            "zoo",
    };
}
