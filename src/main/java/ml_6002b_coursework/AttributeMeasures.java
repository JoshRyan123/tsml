package ml_6002b_coursework;

import experiments.data.DatasetLoading;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.io.IOException;
import java.util.Enumeration;

/**
 * Empty class for Part 2.1 of the coursework.
 */
public class AttributeMeasures {
    // Should follow formula shown in lecture 2, and comment code to indicate edge cases

    // returns info gain for contingency table
    static double measureInformationGainRatio(int[][] table) {
        double Entropy;
        double Bag[];
        double Class[];
        double total = 0;
        double oldEntropy = 0;
        double newEntropy = 0;
        int i, j;

        Bag = new double [table.length];
        Class = new double [table[0].length];
        // iterate through rows (attribute values) in contingency table
        for (i = 0; i < table.length; i++) {
            // iterate through columns (class values) of contingency table
            for (j = 0; j < table[i].length; j++) {
                Bag[i] += table[i][j];
                Class[j] += table[i][j];
                total = total + table[i][j];
            }
        }

        double oldValue = 0;
        int p;
        for (p=0;p<Class.length;p++) {
            oldValue = oldValue + logFunc(Class[p]);
            //System.out.println("log function old value"+logFunc(Class[p]));
        }
        oldEntropy = oldEntropy+(logFunc(total)-oldValue);

        double newValue = 0;
        int k,o;
        for (k=0;k<Bag.length;k++){
            for (o=0;o<Class.length;o++)
                newValue = newValue+logFunc(table[k][o]);
            //System.out.println("log function new value"+logFunc(Bag[k]));
            newValue = newValue-logFunc(Bag[k]);
        }
        newEntropy = newEntropy-newValue;

        Entropy = oldEntropy-newEntropy;

        // Splits with no gain are useless.
        if (Utils.eq(Entropy,0))
            return Double.MAX_VALUE;

        return total/Entropy;
    }
    public static final double logFunc(double num) {
        double log2 = Math.log(2);
        if (num < 1e-6)
            return 0;
        else
            return num*Math.log(num)/log2;
    }


    // returns info gain ratio for contingency table
    static double measureInformationGain(int[][] table) throws Exception {
        // entropy is 1 as 50/50 split (base entropy)
        int k, p;
        double total = 0;
        double [] classCounts = new double[table.length];

        for (k = 0; k < classCounts.length; k++) {
            for (p = 0; p < table[k].length; p++) {
                classCounts[k] = table[0][k]+table[1][k];
                total += table[k][p];
            }
        }

        double baseEntropy = 0;
        for (int j = 0; j < table.length; j++) {
            if (classCounts[j] > 0) {
                baseEntropy -= classCounts[j] * Utils.log2(classCounts[j]);
            }
        }
        baseEntropy /= total;
        baseEntropy += Utils.log2(total);

        System.out.println("starting entropy: "+baseEntropy);

        // set info gain equal to starting entropy
        double infoGain = baseEntropy;
        double probability = 0;
        int i, j;
        // iterate through rows (attribute values) in contingency table
        for (i = 0; i < table.length; i++) {
            // iterate through columns (class values) of contingency table
            for (j = 0; j < table[i].length; j++) {
                probability = (double) (table[i][j]) /
                        ((table[i][0]+table[i][1]));
                infoGain -=  (probability * logFunc(probability));
//                System.out.println("table value [i][0]:"+table[i][0]);
//                System.out.println("table value [i][1]:"+table[i][1]);
//                System.out.println("table value [i][1]+[i][0]:"+(table[i][0]+table[i][1]));
//                System.out.println("probability:"+probability);
//                System.out.println("log function: "+logFunc(probability));
//                System.out.println("infogain: "+infoGain);
            }
        }
        return infoGain;
    }

    // returns gini measure for contingency table
    static double measureGini(int[][] arr) {
        double[] parentDistribution = new double[arr.length];
        double[][] childDistributions = new double[arr.length][2];

        // entropy is 1 as 50/50 split (base entropy)
        int k, p;
        double total = 0;
        double totalLeft = 0;
        double totalRight = 0;
        double [] classCounts = new double[arr.length];

        for (k = 0; k < classCounts.length; k++) {
            for (p = 0; p < arr[k].length; p++) {
                parentDistribution[k] = arr[0][k]+arr[1][k];
                childDistributions[k][p] = arr[k][p];
                total += arr[k][p];
                //totalRight += table[i][0]+table[i][1]
            }
        }
//        for (int q = 0; q < parentDistribution.length; q++) {
//            if (classCounts[q] > 0) {
//                parentDistribution[q] /= total;
//            }
//        }

        double totalWeight = Utils.sum(parentDistribution);
        if (totalWeight==0) return 0;

        double leftWeight = Utils.sum(childDistributions[0]);
        double rightWeight = Utils.sum(childDistributions[1]);

        double parentVal = 0;
        for (int i=0; i<parentDistribution.length; i++) {
            parentVal += (parentDistribution[i]/totalWeight)*(parentDistribution[i]/totalWeight);
        }
        parentVal = 1-parentVal;
        // System.out.println("parent gini index: "+parentVal);

        double leftVal = 0;
        for (int i=0; i<childDistributions[0].length; i++) {
            leftVal += (childDistributions[0][i]/leftWeight)*(childDistributions[0][i]/leftWeight);
        }
        leftVal = 1-leftVal;
        // System.out.println("left gini index: "+leftVal);

        double rightVal = 0;
        for (int i=0; i<childDistributions[1].length; i++) {
            rightVal += (childDistributions[1][i]/rightWeight)*(childDistributions[1][i]/rightWeight);
        }
        rightVal = 1-rightVal;
        // System.out.println("right gini index: "+rightVal);

        return parentVal - leftWeight/totalWeight*leftVal - rightWeight/totalWeight*rightVal;
    }

    // returns chi statistic for contingency table
//    static double measureChiSquared(int[][] arr) {
//        return
//    }


    /**
     * Main method.
     *
     * @param args the options for the attribute measure main
     */
    // main method test harness should test functionality of each split measure
    // should find each measure for the attribute 'Peaty' in terms of the 'diagnosis'.
    public static void main(String[] args) throws Exception {
        System.out.println("Not Implemented.");

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
        for(Instance ins:data){
            Peaty[(int)ins.value(0)][(int)ins.classValue()]++;
        }

        // outputs the split for Peaty: (1):5 and (0):1 on one side (when class value is 0) and
        //                              (0):4 and (1):0 on the other-side (when class value is 1)
        for(int[] x:Peaty) {
            for (int y : x)
                System.out.print(y + ",");
            System.out.print("\n");
        }

        // need to find class probabilities : proportion of each class value in overall data
        // ie : 5/6 and 1/6
        // and: 4/4 and 4/0 (pure)
        // then find the effect (info gain) that node had
        double infoGain = measureInformationGain(Peaty);
        System.out.println(" Measure InfoGain for Peaty = "+1/infoGain);

        double infoGainRatio = measureInformationGainRatio(Peaty);
        System.out.println(" Measure InfoGainRatio for Peaty = "+1/infoGainRatio);

        double giniIndex = measureGini(Peaty);
        System.out.println(" Measure GiniIndex for Peaty = "+giniIndex);



    }

}
