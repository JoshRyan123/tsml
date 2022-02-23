// J48 has the following parameters that can be adjusted.

// (1) binarySplits This specifies whether to use binary splits on nominal data. This is a process by which
//the tree is grown by considering one nominal value versus all other nominal values instead of
//considering a split on each nominal value individually. This results in a tree where there are only two
//branches from any node.

// (2) confidenceFactor This determines how aggressive the pruning process will be. The higher this
//value, the more ‘confident’ you are that the data you are learning from is a good representation of
//all possible events, and therefore the less pruning that will occur. Smaller values induce more
//pruning. This significantly affects classifier performance

// (3) minNumObj This determines what the minimum number of observations are allowed at each leaf
//of the tree. This is another way to control overfitting.

// (4) numFolds This determines how much of the data will be used to prune the tree. One of the folds is
//held out for pruning while the rest grow the tree. The default value of J48 Classifier Parameters 2
//three means one third of the data is used for pruning, while two thirds are used for growing the tree.
//Setting this number too low will increase overfitting.

// (5) unpruned This specifies if the tree should not be pruned.

// (6) reducedErrorPruning Reduced error pruning is an alternative algorithm for pruning that focuses on
//minimizing the statistical error of the tree, instead of the misclassification rate. This is not the
//default pruning mechanism.

// (7) subtreeRaising This is a specific method of pruning whereby a whole set of branches further down
//the tree are moved up to replace branches that were grown above it. This is the default pruning
//mechanism

// (8) useLaplace This applies laplace smoothing to counts at the leaves. This is also sometimes called
//additive smoothing, and is a method by which a certain number is added to all instances in order to
//eliminate circumstances that are statistically undesirable, such as encountering the number zero.
//This is most useful when predicting probabilities

// EXTRA: Explore some of the other decision trees available in Weka experiment and compare to J48.

// building a classifier and looking at the tree.
// 1. Go to the UCI archive, get the balloons and breast cancer data
// 2. Load the data into a single Instances, build a tree on the whole data, then print out the
//    whole tree
// 3. Set binary splits to true, and repeat. Notice the difference in tree (if any!)
// 4. Do the same with reduced error pruning

package Laboratories.lab3;

import experiments.data.DatasetLoading;
import weka.classifiers.Classifier;
import weka.classifiers.Sourcable;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.j48.*;
import weka.core.*;

import java.util.Enumeration;
import java.util.Vector;

// capabilities are inherited from AbstractClassifier, which by default allows all data type
public class C45 implements Classifier, OptionHandler, Drawable, Matchable, Sourcable,
        WeightedInstancesHandler, Summarizable, AdditionalMeasureProducer, CapabilitiesHandler {

    /** for serialization */
    static final long serialVersionUID = -217733168393644444L;


    /** The decision tree */
    private ClassifierTree m_root;


    /** Unpruned tree? */
    private boolean m_unpruned = false;


    /** Confidence level */
    private float m_CF = 0.25f;


    /** Minimum number of instances */
    private int m_minNumObj = 2;


    /** Determines whether probabilities are smoothed using
     Laplace correction when predictions are generated */
    private boolean m_useLaplace = false;


    // Basically builds one of two types of tree, based on whether reduced error pruning is used. The
    // default is to not use it
    private boolean m_reducedErrorPruning = false;


    /** Number of folds for reduced error pruning. */
    private int m_numFolds = 3;


    // If this parameter is false, a node is created for every value of nominal attributes. It defaults to false
    private boolean m_binarySplits = false;


    /** Subtree raising to be performed? */
    private boolean m_subtreeRaising = true;


    /** Cleanup after the tree has been built. */
    private boolean m_noCleanup = false;


    /** Random number seed for reduced-error pruning. */
    private int m_Seed = 1;

    // Tests capabilities here, not in J48.
    // What sort of data does it work for? Have a look.
    // Moving on: the most important method is buildTree
    // On a high level, the m_localModel splits the data into an array of instances called localInstances,
    // creates an array of ClassifierTree offspring (m_sons), then calls getNewTree for each offspring.
    // This is recursive
    // ModelSelection is an abstract class with two subclasses, C45ModelSelection, and BinC45ModelSelection
    // It splits data by attribute and recursively calls for each split (steps 2 and 3 of induction of DT's)
    // C45ModelSelection performs attribute selection and stopping criteria (see labsheet 3)
    public void buildClassifier(Instances instances)
            throws Exception {

        ModelSelection modSelection;

        if (m_binarySplits)
            modSelection = new BinC45ModelSelection(m_minNumObj, instances, true);
        else
            modSelection = new C45ModelSelection(m_minNumObj, instances, true);
        if (!m_reducedErrorPruning)
            // ModelSelection object controls the attribute selection
            m_root = new C45PruneableClassifierTree(modSelection, !m_unpruned, m_CF,
                    m_subtreeRaising, !m_noCleanup, true);
        else
            m_root = new PruneableClassifierTree(modSelection, !m_unpruned, m_numFolds,
                    !m_noCleanup, m_Seed);

        m_root.buildClassifier(instances);
        if (m_binarySplits) {
            ((BinC45ModelSelection)modSelection).cleanup();
        } else {
            ((C45ModelSelection)modSelection).cleanup();
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return m_root.classifyInstance(instance);
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return m_root.distributionForInstance(instance, m_useLaplace);
    }

    // Overriding this method allows you to control what sort of data a classifier can handle,
    // but assume J48 can handle all sorts of input, including missing values
    // Derived classifiers should override this method and first disable all capabilities then
    // enable those that make sense for the scheme
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = new Capabilities(this);
        result.enableAll();

        return result;

//        public Capabilities getCapabilities() {
//            Capabilities      result;
//
//            try {
//                if (!m_reducedErrorPruning)
//                    result = new C45PruneableClassifierTree(null, !m_unpruned, m_CF, m_subtreeRaising, !m_noCleanup).getCapabilities();
//                else
//                    result = new PruneableClassifierTree(null, !m_unpruned, m_numFolds, !m_noCleanup, m_Seed).getCapabilities();
//            }
//            catch (Exception e) {
//                result = new Capabilities(this);
//            }
//
//            result.setOwner(this);
//
//            return result;
//        }
    }

    public static void main(String[] args) throws Exception{
        Instances all;
        String dataPath = "Data/lab1/Arsenal_TEST.arff";
        all = DatasetLoading.loadData(dataPath);

        J48 c45 = new J48();
        System.out.println("Capabilities: " + c45.getCapabilities());
        c45.buildClassifier(all);
        System.out.println("MODEL = " + c45.toString());

//        System.out.println(c45.getReducedErrorPruning());
//        c45.setUnpruned(true);
//        c45.buildClassifier(all);

//        c45.setReducedErrorPruning(true);
//        c45.buildClassifier(all);


    }

    @Override
    public String toSource(String className) throws Exception {
        StringBuffer [] source = m_root.toSource(className);
        return
                "class " + className + " {\n\n"
                        +"  public static double classify(Object[] i)\n"
                        +"    throws Exception {\n\n"
                        +"    double p = Double.NaN;\n"
                        + source[0]  // Assignment code
                        +"    return p;\n"
                        +"  }\n"
                        + source[1]  // Support code
                        +"}\n";
    }

    @Override
    public Enumeration<String> enumerateMeasures() {
        Vector newVector = new Vector(3);
        newVector.addElement("measureTreeSize");
        newVector.addElement("measureNumLeaves");
        newVector.addElement("measureNumRules");
        return newVector.elements();
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

    @Override
    public double getMeasure(String additionalMeasureName) {
        if (additionalMeasureName.compareToIgnoreCase("measureNumRules") == 0) {
            return measureNumRules();
        } else if (additionalMeasureName.compareToIgnoreCase("measureTreeSize") == 0) {
            return measureTreeSize();
        } else if (additionalMeasureName.compareToIgnoreCase("measureNumLeaves") == 0) {
            return measureNumLeaves();
        } else {
            throw new IllegalArgumentException(additionalMeasureName
                    + " not supported (j48)");
        }
    }

    @Override
    public int graphType() {
        return Drawable.TREE;
    }

    @Override
    public String graph() throws Exception {
        return m_root.graph();
    }

    @Override
    public String prefix() throws Exception {
        return m_root.prefix();
    }

    // vectorize elements
    @Override
    public Enumeration<Option> listOptions() {
        Vector newVector = new Vector(9);

        newVector.
                addElement(new Option("\tUse unpruned tree.",
                        "U", 0, "-U"));
        newVector.
                addElement(new Option("\tSet confidence threshold for pruning.\n" +
                        "\t(default 0.25)",
                        "C", 1, "-C <pruning confidence>"));
        newVector.
                addElement(new Option("\tSet minimum number of instances per leaf.\n" +
                        "\t(default 2)",
                        "M", 1, "-M <minimum number of instances>"));
        newVector.
                addElement(new Option("\tUse reduced error pruning.",
                        "R", 0, "-R"));
        newVector.
                addElement(new Option("\tSet number of folds for reduced error\n" +
                        "\tpruning. One fold is used as pruning set.\n" +
                        "\t(default 3)",
                        "N", 1, "-N <number of folds>"));
        newVector.
                addElement(new Option("\tUse binary splits only.",
                        "B", 0, "-B"));
        newVector.
                addElement(new Option("\tDon't perform subtree raising.",
                        "S", 0, "-S"));
        newVector.
                addElement(new Option("\tDo not clean up after the tree has been built.",
                        "L", 0, "-L"));
        newVector.
                addElement(new Option("\tLaplace smoothing for predicted probabilities.",
                        "A", 0, "-A"));
        newVector.
                addElement(new Option("\tSeed for random data shuffling (default 1).",
                        "Q", 1, "-Q <seed>"));

        return newVector.elements();
    }

    public float getConfidenceFactor() {

        return m_CF;
    }
    @Override
    public void setOptions(String[] options) throws Exception {

        // Other options
        String minNumString = Utils.getOption('M', options);
        if (minNumString.length() != 0) {
            m_minNumObj = Integer.parseInt(minNumString);
        } else {
            m_minNumObj = 2;
        }
        m_binarySplits = Utils.getFlag('B', options);
        m_useLaplace = Utils.getFlag('A', options);

        // Pruning options
        m_unpruned = Utils.getFlag('U', options);
        m_subtreeRaising = !Utils.getFlag('S', options);
        m_noCleanup = Utils.getFlag('L', options);
        if ((m_unpruned) && (!m_subtreeRaising)) {
            throw new Exception("Subtree raising doesn't need to be unset for unpruned tree!");
        }
        m_reducedErrorPruning = Utils.getFlag('R', options);
        if ((m_unpruned) && (m_reducedErrorPruning)) {
            throw new Exception("Unpruned tree and reduced error pruning can't be selected " +
                    "simultaneously!");
        }
        String confidenceString = Utils.getOption('C', options);
        if (confidenceString.length() != 0) {
            if (m_reducedErrorPruning) {
                throw new Exception("Setting the confidence doesn't make sense " +
                        "for reduced error pruning.");
            } else if (m_unpruned) {
                throw new Exception("Doesn't make sense to change confidence for unpruned "
                        +"tree!");
            } else {
//                m_CF = (new Float(confidenceString)).floatValue();
                if ((m_CF <= 0) || (m_CF >= 1)) {
                    throw new Exception("Confidence has to be greater than zero and smaller " +
                            "than one!");
                }
            }
        } else {
            m_CF = 0.25f;
        }
        String numFoldsString = Utils.getOption('N', options);
        if (numFoldsString.length() != 0) {
            if (!m_reducedErrorPruning) {
                throw new Exception("Setting the number of folds" +
                        " doesn't make sense if" +
                        " reduced error pruning is not selected.");
            } else {
                m_numFolds = Integer.parseInt(numFoldsString);
            }
        } else {
            m_numFolds = 3;
        }
        String seedString = Utils.getOption('Q', options);
        if (seedString.length() != 0) {
            m_Seed = Integer.parseInt(seedString);
        } else {
            m_Seed = 1;
        }
    }

    @Override
    public String[] getOptions() {
        String [] options = new String [14];
        int current = 0;

        if (m_noCleanup) {
            options[current++] = "-L";
        }
        if (m_unpruned) {
            options[current++] = "-U";
        } else {
            if (!m_subtreeRaising) {
                options[current++] = "-S";
            }
            if (m_reducedErrorPruning) {
                options[current++] = "-R";
                options[current++] = "-N"; options[current++] = "" + m_numFolds;
                options[current++] = "-Q"; options[current++] = "" + m_Seed;
            } else {
                options[current++] = "-C"; options[current++] = "" + m_CF;
            }
        }
        if (m_binarySplits) {
            options[current++] = "-B";
        }
        options[current++] = "-M"; options[current++] = "" + m_minNumObj;
        if (m_useLaplace) {
            options[current++] = "-A";
        }

        while (current < options.length) {
            options[current++] = "";
        }
        return options;
    }

    @Override
    public String toSummaryString() {
        return "Number of leaves: " + m_root.numLeaves() + "\n"
                + "Size of the tree: " + m_root.numNodes() + "\n";
    }

    public String toString() {

        if (m_root == null) {
            return "No classifier built";
        }
        if (m_unpruned)
            return "J48 unpruned tree\n------------------\n" + m_root.toString();
        else
            return "J48 pruned tree\n------------------\n" + m_root.toString();
    }
}
