//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by FernFlower decompiler)
//

package Laboratories.lab3;

import experiments.data.DatasetLoading;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Sourcable;
import weka.classifiers.trees.j48.*;
import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

public class J48 extends AbstractClassifier implements OptionHandler, Drawable, Matchable, Sourcable, WeightedInstancesHandler, Summarizable, AdditionalMeasureProducer, TechnicalInformationHandler, PartitionGenerator {
    static final long serialVersionUID = -217733168393644444L;
    protected ClassifierTree m_root;
    protected boolean m_unpruned = false;
    protected boolean m_collapseTree = true;
    protected float m_CF = 0.25F;
    protected int m_minNumObj = 2;
    protected boolean m_useMDLcorrection = true;
    protected boolean m_useLaplace = false;
    protected boolean m_reducedErrorPruning = false;
    protected int m_numFolds = 3;
    protected boolean m_binarySplits = false;
    protected boolean m_subtreeRaising = true;
    protected boolean m_noCleanup = false;
    protected int m_Seed = 1;
    protected boolean m_doNotMakeSplitPointActualValue;

    public J48() {
    }

    public String globalInfo() {
        return "Class for generating a pruned or unpruned C4.5 decision tree. For more information, see\n\n" + this.getTechnicalInformation().toString();
    }

    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(Type.BOOK);
        result.setValue(Field.AUTHOR, "Ross Quinlan");
        result.setValue(Field.YEAR, "1993");
        result.setValue(Field.TITLE, "C4.5: Programs for Machine Learning");
        result.setValue(Field.PUBLISHER, "Morgan Kaufmann Publishers");
        result.setValue(Field.ADDRESS, "San Mateo, CA");
        return result;
    }

    public Capabilities getCapabilities() {
        Capabilities result = new Capabilities(this);
        result.disableAll();
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capability.DATE_ATTRIBUTES);
        result.enable(Capability.MISSING_VALUES);
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);
        result.setMinimumNumberInstances(0);
        return result;
    }

    public void buildClassifier(Instances instances) throws Exception {
        Object modSelection;
        // Mimics behaviour of CART Decision Tree
        if (this.m_binarySplits) {
            // uses enumeration
            modSelection = new BinC45ModelSelection(this.m_minNumObj, instances, this.m_useMDLcorrection);
        } else {
            modSelection = new C45ModelSelection(this.m_minNumObj, instances, this.m_useMDLcorrection);
        }

        if (!this.m_reducedErrorPruning) {
            this.m_root = new C45PruneableClassifierTree((ModelSelection)modSelection, !this.m_unpruned, this.m_CF, this.m_subtreeRaising, !this.m_noCleanup, this.m_collapseTree);
        } else {
            this.m_root = new PruneableClassifierTree((ModelSelection)modSelection, !this.m_unpruned, this.m_numFolds, !this.m_noCleanup, this.m_Seed);
        }

        this.m_root.buildClassifier(instances);
        if (this.m_binarySplits) {
            ((BinC45ModelSelection)modSelection).cleanup();
        } else {
            ((C45ModelSelection)modSelection).cleanup();
        }

    }

    public double classifyInstance(Instance instance) throws Exception {
        return this.m_root.classifyInstance(instance);
    }

    public final double[] distributionForInstance(Instance instance) throws Exception {
        return this.m_root.distributionForInstance(instance, this.m_useLaplace);
    }

    public int graphType() {
        return 1;
    }

    public String graph() throws Exception {
        return this.m_root.graph();
    }

    public String prefix() throws Exception {
        return this.m_root.prefix();
    }

    public String toSource(String className) throws Exception {
        StringBuffer[] source = this.m_root.toSource(className);
        return "class " + className + " {\n\n  public static double classify(Object[] i)\n    throws Exception {\n\n    double p = Double.NaN;\n" + source[0] + "    return p;\n  }\n" + source[1] + "}\n";
    }

    public Enumeration<Option> listOptions() {
        Vector<Option> newVector = new Vector(13);
        newVector.addElement(new Option("\tUse unpruned tree.", "U", 0, "-U"));
        newVector.addElement(new Option("\tDo not collapse tree.", "O", 0, "-O"));
        newVector.addElement(new Option("\tSet confidence threshold for pruning.\n\t(default 0.25)", "C", 1, "-C <pruning confidence>"));
        newVector.addElement(new Option("\tSet minimum number of instances per leaf.\n\t(default 2)", "M", 1, "-M <minimum number of instances>"));
        newVector.addElement(new Option("\tUse reduced error pruning.", "R", 0, "-R"));
        newVector.addElement(new Option("\tSet number of folds for reduced error\n\tpruning. One fold is used as pruning set.\n\t(default 3)", "N", 1, "-N <number of folds>"));
        newVector.addElement(new Option("\tUse binary splits only.", "B", 0, "-B"));
        newVector.addElement(new Option("\tDo not perform subtree raising.", "S", 0, "-S"));
        newVector.addElement(new Option("\tDo not clean up after the tree has been built.", "L", 0, "-L"));
        newVector.addElement(new Option("\tLaplace smoothing for predicted probabilities.", "A", 0, "-A"));
        newVector.addElement(new Option("\tDo not use MDL correction for info gain on numeric attributes.", "J", 0, "-J"));
        newVector.addElement(new Option("\tSeed for random data shuffling (default 1).", "Q", 1, "-Q <seed>"));
        newVector.addElement(new Option("\tDo not make split point actual value.", "-doNotMakeSplitPointActualValue", 0, "-doNotMakeSplitPointActualValue"));
        newVector.addAll(Collections.list(super.listOptions()));
        return newVector.elements();
    }

    public void setOptions(String[] options) throws Exception {
        String minNumString = Utils.getOption('M', options);
        if (minNumString.length() != 0) {
            this.m_minNumObj = Integer.parseInt(minNumString);
        } else {
            this.m_minNumObj = 2;
        }

        this.m_binarySplits = Utils.getFlag('B', options);
        this.m_useLaplace = Utils.getFlag('A', options);
        this.m_useMDLcorrection = !Utils.getFlag('J', options);
        this.m_unpruned = Utils.getFlag('U', options);
        this.m_collapseTree = !Utils.getFlag('O', options);
        this.m_subtreeRaising = !Utils.getFlag('S', options);
        this.m_noCleanup = Utils.getFlag('L', options);
        this.m_doNotMakeSplitPointActualValue = Utils.getFlag("doNotMakeSplitPointActualValue", options);
        if (this.m_unpruned && !this.m_subtreeRaising) {
            throw new Exception("Subtree raising doesn't need to be unset for unpruned tree!");
        } else {
            this.m_reducedErrorPruning = Utils.getFlag('R', options);
            if (this.m_unpruned && this.m_reducedErrorPruning) {
                throw new Exception("Unpruned tree and reduced error pruning can't be selected simultaneously!");
            } else {
                String confidenceString = Utils.getOption('C', options);
                if (confidenceString.length() != 0) {
                    if (this.m_reducedErrorPruning) {
                        throw new Exception("Setting the confidence doesn't make sense for reduced error pruning.");
                    }

                    if (this.m_unpruned) {
                        throw new Exception("Doesn't make sense to change confidence for unpruned tree!");
                    }

//                    this.m_CF = new Float(confidenceString).floatValue();
                    if (this.m_CF <= 0.0F || this.m_CF >= 1.0F) {
                        throw new Exception("Confidence has to be greater than zero and smaller than one!");
                    }
                } else {
                    this.m_CF = 0.25F;
                }

                String numFoldsString = Utils.getOption('N', options);
                if (numFoldsString.length() != 0) {
                    if (!this.m_reducedErrorPruning) {
                        throw new Exception("Setting the number of folds doesn't make sense if reduced error pruning is not selected.");
                    }

                    this.m_numFolds = Integer.parseInt(numFoldsString);
                } else {
                    this.m_numFolds = 3;
                }

                String seedString = Utils.getOption('Q', options);
                if (seedString.length() != 0) {
                    this.m_Seed = Integer.parseInt(seedString);
                } else {
                    this.m_Seed = 1;
                }

                super.setOptions(options);
                Utils.checkForRemainingOptions(options);
            }
        }
    }

    public String[] getOptions() {
        Vector<String> options = new Vector();
        if (this.m_noCleanup) {
            options.add("-L");
        }

        if (!this.m_collapseTree) {
            options.add("-O");
        }

        if (this.m_unpruned) {
            options.add("-U");
        } else {
            if (!this.m_subtreeRaising) {
                options.add("-S");
            }

            if (this.m_reducedErrorPruning) {
                options.add("-R");
                options.add("-N");
                options.add("" + this.m_numFolds);
                options.add("-Q");
                options.add("" + this.m_Seed);
            } else {
                options.add("-C");
                options.add("" + this.m_CF);
            }
        }

        if (this.m_binarySplits) {
            options.add("-B");
        }

        options.add("-M");
        options.add("" + this.m_minNumObj);
        if (this.m_useLaplace) {
            options.add("-A");
        }

        if (!this.m_useMDLcorrection) {
            options.add("-J");
        }

        if (this.m_doNotMakeSplitPointActualValue) {
            options.add("-doNotMakeSplitPointActualValue");
        }

        Collections.addAll(options, super.getOptions());
        return (String[])options.toArray(new String[0]);
    }

    public String seedTipText() {
        return "The seed used for randomizing the data when reduced-error pruning is used.";
    }

    public int getSeed() {
        return this.m_Seed;
    }

    public void setSeed(int newSeed) {
        this.m_Seed = newSeed;
    }

    public String useLaplaceTipText() {
        return "Whether counts at leaves are smoothed based on Laplace.";
    }

    public boolean getUseLaplace() {
        return this.m_useLaplace;
    }

    public void setUseLaplace(boolean newuseLaplace) {
        this.m_useLaplace = newuseLaplace;
    }

    public String useMDLcorrectionTipText() {
        return "Whether MDL correction is used when finding splits on numeric attributes.";
    }

    public boolean getUseMDLcorrection() {
        return this.m_useMDLcorrection;
    }

    public void setUseMDLcorrection(boolean newuseMDLcorrection) {
        this.m_useMDLcorrection = newuseMDLcorrection;
    }

    public String toString() {
        if (this.m_root == null) {
            return "No classifier built";
        } else {
            return this.m_unpruned ? "J48 unpruned tree\n------------------\n" + this.m_root.toString() : "J48 pruned tree\n------------------\n" + this.m_root.toString();
        }
    }

    public String toSummaryString() {
        return "Number of leaves: " + this.m_root.numLeaves() + "\nSize of the tree: " + this.m_root.numNodes() + "\n";
    }

    public double measureTreeSize() {
        return (double)this.m_root.numNodes();
    }

    public double measureNumLeaves() {
        return (double)this.m_root.numLeaves();
    }

    public double measureNumRules() {
        return (double)this.m_root.numLeaves();
    }

    public Enumeration<String> enumerateMeasures() {
        Vector<String> newVector = new Vector(3);
        newVector.addElement("measureTreeSize");
        newVector.addElement("measureNumLeaves");
        newVector.addElement("measureNumRules");
        return newVector.elements();
    }

    public double getMeasure(String additionalMeasureName) {
        if (additionalMeasureName.compareToIgnoreCase("measureNumRules") == 0) {
            return this.measureNumRules();
        } else if (additionalMeasureName.compareToIgnoreCase("measureTreeSize") == 0) {
            return this.measureTreeSize();
        } else if (additionalMeasureName.compareToIgnoreCase("measureNumLeaves") == 0) {
            return this.measureNumLeaves();
        } else {
            throw new IllegalArgumentException(additionalMeasureName + " not supported (j48)");
        }
    }

    public String unprunedTipText() {
        return "Whether pruning is performed.";
    }

    public boolean getUnpruned() {
        return this.m_unpruned;
    }

    public void setUnpruned(boolean v) {
        if (v) {
            this.m_reducedErrorPruning = false;
        }

        this.m_unpruned = v;
    }

    public String collapseTreeTipText() {
        return "Whether parts are removed that do not reduce training error.";
    }

    public boolean getCollapseTree() {
        return this.m_collapseTree;
    }

    public void setCollapseTree(boolean v) {
        this.m_collapseTree = v;
    }

    public String confidenceFactorTipText() {
        return "The confidence factor used for pruning (smaller values incur more pruning).";
    }

    public float getConfidenceFactor() {
        return this.m_CF;
    }

    public void setConfidenceFactor(float v) {
        this.m_CF = v;
    }

    public String minNumObjTipText() {
        return "The minimum number of instances per leaf.";
    }

    public int getMinNumObj() {
        return this.m_minNumObj;
    }

    public void setMinNumObj(int v) {
        this.m_minNumObj = v;
    }

    public String reducedErrorPruningTipText() {
        return "Whether reduced-error pruning is used instead of C.4.5 pruning.";
    }

    public boolean getReducedErrorPruning() {
        return this.m_reducedErrorPruning;
    }

    public void setReducedErrorPruning(boolean v) {
        if (v) {
            this.m_unpruned = false;
        }

        this.m_reducedErrorPruning = v;
    }

    public String numFoldsTipText() {
        return "Determines the amount of data used for reduced-error pruning.  One fold is used for pruning, the rest for growing the tree.";
    }

    public int getNumFolds() {
        return this.m_numFolds;
    }

    public void setNumFolds(int v) {
        this.m_numFolds = v;
    }

    public String binarySplitsTipText() {
        return "Whether to use binary splits on nominal attributes when building the trees.";
    }

    public boolean getBinarySplits() {
        return this.m_binarySplits;
    }

    public void setBinarySplits(boolean v) {
        this.m_binarySplits = v;
    }

    public String subtreeRaisingTipText() {
        return "Whether to consider the subtree raising operation when pruning.";
    }

    public boolean getSubtreeRaising() {
        return this.m_subtreeRaising;
    }

    public void setSubtreeRaising(boolean v) {
        this.m_subtreeRaising = v;
    }

    public String saveInstanceDataTipText() {
        return "Whether to save the training data for visualization.";
    }

    public boolean getSaveInstanceData() {
        return this.m_noCleanup;
    }

    public void setSaveInstanceData(boolean v) {
        this.m_noCleanup = v;
    }

    public String doNotMakeSplitPointActualValueTipText() {
        return "If true, the split point is not relocated to an actual data value. This can yield substantial speed-ups for large datasets with numeric attributes.";
    }

    public boolean getDoNotMakeSplitPointActualValue() {
        return this.m_doNotMakeSplitPointActualValue;
    }

    public void setDoNotMakeSplitPointActualValue(boolean m_doNotMakeSplitPointActualValue) {
        this.m_doNotMakeSplitPointActualValue = m_doNotMakeSplitPointActualValue;
    }

    public String getRevision() {
        return RevisionUtils.extract("$Revision: 11194 $");
    }

    public void generatePartition(Instances data) throws Exception {
        this.buildClassifier(data);
    }

    public double[] getMembershipValues(Instance inst) throws Exception {
        return this.m_root.getMembershipValues(inst);
    }

    public int numElements() throws Exception {
        return this.m_root.numNodes();
    }

    public static void main(String[] argv) throws Exception {

        Instances all;
        String dataPath = "Data/lab1/Aedes_Female_VS_House_Fly_POWER.arff";
        all = DatasetLoading.loadData(dataPath);
        //Build on all the iris data
        J48 c45 = new J48();
//        weka.classifiers.trees.J48 C45 = new weka.classifiers.trees.J48();

        c45.buildClassifier(all);
//        C45.buildClassifier(all);

        System.out.println(c45.getCapabilities());
        System.out.println(c45);

//        System.out.println(C45.getCapabilities());
//        System.out.println(C45);



        System.out.println(c45.getReducedErrorPruning());

        c45.setUnpruned(true);

        c45.setReducedErrorPruning(true);

        c45.buildClassifier(all);

        System.out.println(c45);

        JoshEnsemble ens = new JoshEnsemble();
        ens.buildClassifier(all);



    }
}
