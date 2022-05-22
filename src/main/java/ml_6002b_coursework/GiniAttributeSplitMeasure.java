package ml_6002b_coursework;

import experiments.data.DatasetLoading;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Enumeration;

//  Implement and test the skeleton class IGAttributeSplitMeasure
public class GiniAttributeSplitMeasure extends AttributeSplitMeasure {
    // The choice of measure should be controlled by a boolean variable called useGain
    private boolean useGain;

    // split is performed using information gain or information gain ratio
    @Override
    public double computeAttributeQuality(Instances data, Attribute att) throws Exception {
        AttributeMeasures am = new AttributeMeasures();

        // cannot split data on single valued attribute
        if (att.numValues() <= 1) {
            return 0.0;
        }

        int[][] table = new int[(int) att.numValues()][data.numClasses()];

        int k;
        int trueAttIndex = 0;
        for (k = 0; k < data.numAttributes(); k++) {
            if (data.attribute(k).name()==att.name()) {
                trueAttIndex = k;
                //System.out.println("true value of index (index of attribute passed): "+k);
            }
        }
        for(Instance ins:data){
            table[(int)ins.value(trueAttIndex)][(int)ins.classValue()]++;
        }
//        for(int[] x:table) {
//            for (int y : x)
//                System.out.print(" "+y + ", ");
//            System.out.print("\n");
//        }

        double measure = am.measureGini(table);

        return measure;
    }

    /**
     * Main method.
     *
     * @param args the options for the split measure main
     */
    public static void main(String[] args) throws Exception {
        Instances data = DatasetLoading.loadData("src\\main\\java\\ml_6002b_coursework\\test_data\\Whiskey.arff");

        GiniAttributeSplitMeasure gi = new GiniAttributeSplitMeasure();

        double[] impurityMeasures = new double[data.numAttributes()-1];
        Enumeration attEnum = data.enumerateAttributes();
        while (attEnum.hasMoreElements()) {
            Attribute att = (Attribute) attEnum.nextElement();
            impurityMeasures[att.index()] = gi.computeAttributeQuality(data, att);
            System.out.println("measure Gini Index for attribute " +att.name()+ " splitting diagnosis = "+impurityMeasures[att.index()]);
        }
    }
}