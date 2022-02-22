package Laboratories.lab3;

//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by FernFlower decompiler)
//
//basically counts how
//many cases there are of each class
//
//Using the class value to index into
//an array is a common pattern
//
//

import tsml.src.main.java.experiments.data.DatasetLoading;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Sourcable;
import weka.core.*;
import weka.core.Capabilities.Capability;

import java.util.Iterator;

public class ZeroR extends AbstractClassifier implements WeightedInstancesHandler, Sourcable {
    static final long serialVersionUID = 48055541465867954L;
    private double m_ClassValue;
    // We are focussing only on the
    //scenario where class attribute is NOMINAL. You can ignore other code
    private double[] m_Counts;
    private Attribute m_Class;

    public ZeroR() {
    }

    public String globalInfo() {
        return "Class for building and using a 0-R classifier. Predicts the mean (for a numeric class) or the mode (for a nominal class).";
    }

    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        // only interested in nominal classes, so this is the bit that
        //counts the number of each class
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        // if class value is numeric the problem is regression and not classification
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capability.DATE_ATTRIBUTES);
        result.enable(Capability.STRING_ATTRIBUTES);
        result.enable(Capability.RELATIONAL_ATTRIBUTES);
        result.enable(Capability.MISSING_VALUES);
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.NUMERIC_CLASS);
        result.enable(Capability.DATE_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);
        result.setMinimumNumberInstances(0);
        return result;
    }

    public void buildClassifier(Instances instances) throws Exception {
        this.getCapabilities().testWithFail(instances);

        double sumOfWeights = 0.0D;

        this.m_Class = instances.classAttribute();
        this.m_ClassValue = 0.0D;

        switch(instances.classAttribute().type()) {
            case 0:
                this.m_Counts = null;
                break;
            case 1:
                this.m_Counts = new double[instances.numClasses()];

                for(int i = 0; i < this.m_Counts.length; ++i) {
                    this.m_Counts[i] = 1.0D;
                }

                sumOfWeights = (double)instances.numClasses();
        }

        Iterator var8 = instances.iterator();

        while(var8.hasNext()) {
            Instance instance = (Instance)var8.next();
            double classValue = instance.classValue();
            // might be not needed
            if (!Utils.isMissingValue(classValue)) {
                // This code basically counts how
                //many cases there are of each class. Using the class value to index into
                //an array is a common pattern
                if (instances.classAttribute().isNominal()) {
                    double[] var10000 = this.m_Counts;
                    var10000[(int)classValue] += instance.weight();
                } else {
                    this.m_ClassValue += instance.weight() * classValue;
                }

                sumOfWeights += instance.weight();
            }
        }

        if (instances.classAttribute().isNumeric()) {
            if (Utils.gr(sumOfWeights, 0.0D)) {
                this.m_ClassValue /= sumOfWeights;
            }
        } else {
            //  This sets the majority class, and normalises the counts to be
            //probabilities
            this.m_ClassValue = (double)Utils.maxIndex(this.m_Counts);
            Utils.normalize(this.m_Counts, sumOfWeights);
        }

    }


    //  This is the prediction stage
    public double classifyInstance(Instance instance) {
        return this.m_ClassValue;
    }
    //  This is the prediction stage
    public double[] distributionForInstance(Instance instance) throws Exception {
        if (this.m_Counts == null) {
            double[] result = new double[]{this.m_ClassValue};
            return result;
        } else {
            return (double[])this.m_Counts.clone();
        }
    }

    public String toSource(String className) throws Exception {
        StringBuffer result = new StringBuffer();
        result.append("class " + className + " {\n");
        result.append("  public static double classify(Object[] i) {\n");
        if (this.m_Counts != null) {
            result.append("    // always predicts label '" + this.m_Class.value((int)this.m_ClassValue) + "'\n");
        }

        result.append("    return " + this.m_ClassValue + ";\n");
        result.append("  }\n");
        result.append("}\n");
        return result.toString();
    }

    public String toString() {
        if (this.m_Class == null) {
            return "ZeroR: No model built yet.";
        } else {
            return this.m_Counts == null ? "ZeroR predicts class value: " + this.m_ClassValue : "ZeroR predicts class value: " + this.m_Class.value((int)this.m_ClassValue);
        }
    }

    public String getRevision() {
        return RevisionUtils.extract("$Revision: 12024 $");
    }

    public static void main(String[] argv) throws Exception {
//        runClassifier(new weka.classifiers.rules.ZeroR(), argv);

        ZeroR zero = new ZeroR();

        Instances all;
        String dataPath = "Data/lab1/Arsenal_TEST.arff";
        all = DatasetLoading.loadData(dataPath);

        zero.buildClassifier(all);
        System.out.println(zero.toSource(dataPath));
    }
}