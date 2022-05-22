package ml_6002b_coursework;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.NumericToNominal;

/**
 * Interface for alternative attribute split measures for Part 2.2 of the coursework
 */
public abstract class AttributeSplitMeasure {

    public abstract double computeAttributeQuality(Instances data, Attribute att) throws Exception;

    // An instance filter that discretizes a range of numeric attributes in the dataset into nominal attributes based
    // on provided split value.
    // Discretization by binning.
    public Instances splitDataOnNumeric(Instances data, Attribute att, double splitValue) throws Exception {
        // System.out.println(data);

        // find attribute index in the dataset
        int trueAttIndex = 0;
        // class attribute removed to avoid making discrete
        for (int k = 0; k < data.numAttributes()-1; k++) {
            if (data.attribute(k).name()==att.name()) {
                trueAttIndex = k;
            }
        }

        // iterate through instances in data binning values for the given attribute
        for(Instance ins:data){
            // above split value (1)
            if (ins.value(trueAttIndex) >= splitValue) {
                ins.setValue(trueAttIndex, 1);
            }
            else if (ins.value(trueAttIndex) < splitValue) {
                // below split value (0)
                ins.setValue(trueAttIndex, 0);
            }
            else {
                break;
            }
        }

        System.out.println(data);

        return data;
    }

    // if split value is unknown find average of attributes numeric values - (using mean as split value)
    public Instances splitDataOnNumeric(Instances data, Attribute att) throws Exception {
        // System.out.println(data);

        // find attribute index in the dataset
        int trueAttIndex = 0;
        // class attribute removed to avoid making discrete
        for (int k = 0; k < data.numAttributes()-1; k++) {
            if (data.attribute(k).name()==att.name()) {
                trueAttIndex = k;
            }
        }

        int totalSumAttValues = 0;
        //iterate through instances in data binning values for the given attribute
        for (Instance inst: data) {
            inst.value(trueAttIndex);
            totalSumAttValues += inst.value(trueAttIndex);
            //System.out.println("this attributes instance value for instance: "+inst+" ///// "+inst.value(trueAttIndex));
        }
        //System.out.println("\n\n\n\n\n total for attribute " + trueAttIndex +":"+ totalSumAttValues);
        //System.out.println("splitValue: "+ totalSumAttValues/data.numInstances()+"\n\n\n\n\n\n");

        // find average value seen for the split value
        int splitValue = totalSumAttValues/data.numInstances();

        // iterate through instances in data binning values for the given attribute
        for(Instance ins:data){
            // above split value (1)
            if (ins.value(trueAttIndex) >= splitValue) {
                //System.out.println("\n\n\n instance value: "+ins.value(trueAttIndex));
                //System.out.println("attribute index value: "+trueAttIndex);
                //System.out.println("splitValue: "+splitValue);
                ins.setValue(trueAttIndex, 1);
                // System.out.println("Value after binning: "+ 1);

            }
            // below split value (0)
            else if (ins.value(trueAttIndex) < splitValue) {
                //System.out.println("\n\n\ninstance value: "+ins.value(trueAttIndex));
                //System.out.println("attribute index value: "+trueAttIndex);
                //System.out.println("splitValue: "+splitValue);
                ins.setValue(trueAttIndex, 0);
                //System.out.println("Value after binning: "+ 0);
            }
            else {
                break;
            }
            // System.out.println("Value after binning should be: "+ (ins.value(trueAttIndex) > splitValue)+"\n\n\n");
        }

        //System.out.println(data);

        return data;
    }

    /**
     * Splits a dataset according to the values of a nominal attribute.
     *
     * @param data the data which is to be split
     * @param att the attribute to be used for splitting
     * @return the sets of instances produced by the split
     */
    public Instances[] splitData(Instances data, Attribute att) throws Exception {
        Instances[] splitData = new Instances[att.numValues()];
        for (int i = 0; i < att.numValues(); i++) {
            splitData[i] = new Instances(data, data.numInstances());
        }

        for (Instance inst: data) {
            splitData[(int) inst.value(att)].add(inst);
            //System.out.println(att);
        }

        for (Instances split : splitData) {
            split.compactify();
        }

        return splitData;
    }

}