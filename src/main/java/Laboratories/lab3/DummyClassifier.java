package Laboratories.lab3;

import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instances;

public class DummyClassifier extends AbstractClassifier {
    double[] classDistribution;

    @Override
    public Capabilities getCapabilities(){
        return null;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        getCapabilities().testWithFail(data);
    }

//    public double[] distributionForInstance(Instance ins) {
//        return classDistribution;
//    }
}
//public static enum Capability {
//    NOMINAL_ATTRIBUTES(5, "Nominal attributes"),
//    BINARY_ATTRIBUTES(5, "Binary attributes"),
//    UNARY_ATTRIBUTES(5, "Unary attributes"),
//    EMPTY_NOMINAL_ATTRIBUTES(5, "Empty nominal attributes"),
//    NUMERIC_ATTRIBUTES(5, "Numeric attributes"),
//    DATE_ATTRIBUTES(5, "Date attributes"),
//    STRING_ATTRIBUTES(5, "String attributes"),
//    RELATIONAL_ATTRIBUTES(5, "Relational attributes"),
//    MISSING_VALUES(4, "Missing values"),
//    NO_CLASS(8, "No class"),
//    NOMINAL_CLASS(10, "Nominal class"),
//    BINARY_CLASS(10, "Binary class"),
//    UNARY_CLASS(10, "Unary class"),
//    EMPTY_NOMINAL_CLASS(10, "Empty nominal class"),
//    NUMERIC_CLASS(10, "Numeric class"),
//    DATE_CLASS(10, "Date class"),
//    STRING_CLASS(10, "String class"),
//    RELATIONAL_CLASS(10, "Relational class"),
//    MISSING_CLASS_VALUES(8, "Missing class values"),
//    ONLY_MULTIINSTANCE(16, "Only multi-Instance data");
//
//    private int m_Flags = 0;
//    private String m_Display;
//
//    private Capability(int flags, String display) {
//        this.m_Flags = flags;
//        this.m_Display = display;
//    }
//
//    public boolean isAttribute() {
//        return (this.m_Flags & 1) == 1;
//    }
//
//    public boolean isClass() {
//        return (this.m_Flags & 2) == 2;
//    }
//
//    public boolean isAttributeCapability() {
//        return (this.m_Flags & 4) == 4;
//    }
//
//    public boolean isOtherCapability() {
//        return (this.m_Flags & 16) == 16;
//    }
//
//    public boolean isClassCapability() {
//        return (this.m_Flags & 8) == 8;
//    }
//
//    public String toString() {
//        return this.m_Display;
//    }
//}