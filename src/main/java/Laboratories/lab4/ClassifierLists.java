package Laboratories.lab4;

//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by FernFlower decompiler)
//

import tsml.classifiers.dictionary_based.BagOfPatterns;
import tsml.classifiers.distance_based.ProximityForestWrapper;
import tsml.classifiers.distance_based.knn.KNNLOOCV;
import tsml.classifiers.frequency_based.RISE;
import tsml.classifiers.hybrids.TSCHIEFWrapper;
import tsml.src.main.java.evaluation.tuning.ParameterSpace;
import tsml.src.main.java.experiments.Experiments.ExperimentalArguments;
import tsml.src.main.java.machine_learning.classifiers.PLSNominalClassifier;
import tsml.src.main.java.machine_learning.classifiers.ensembles.CAWPE;
import tsml.src.main.java.machine_learning.classifiers.kNN;
import tsml.src.main.java.machine_learning.classifiers.tuned.TunedClassifier;
import tsml.src.main.java.machine_learning.classifiers.tuned.TunedXGBoost;
import tsml.src.main.java.tsml.classifiers.dictionary_based.*;
import tsml.src.main.java.tsml.classifiers.dictionary_based.boss_variants.BOSSC45;
import tsml.src.main.java.tsml.classifiers.dictionary_based.boss_variants.BoTSWEnsemble;
import tsml.src.main.java.tsml.classifiers.distance_based.*;
import tsml.src.main.java.tsml.classifiers.distance_based.elastic_ensemble.ElasticEnsemble;
import tsml.src.main.java.tsml.classifiers.hybrids.HIVE_COTE;
import tsml.src.main.java.tsml.classifiers.interval_based.LPS;
import tsml.src.main.java.tsml.classifiers.interval_based.TSF;
import tsml.src.main.java.tsml.classifiers.legacy.COTE.FlatCote;
import tsml.src.main.java.tsml.classifiers.legacy.COTE.HiveCote;
import tsml.src.main.java.tsml.classifiers.multivariate.*;
import tsml.src.main.java.tsml.classifiers.shapelet_based.FastShapelets;
import tsml.src.main.java.tsml.classifiers.shapelet_based.LearnShapelets;
import tsml.src.main.java.tsml.classifiers.shapelet_based.ShapeletTransformClassifier;
import tsml.src.main.java.tsml.classifiers.shapelet_based.ShapeletTree;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.EuclideanDistance;
import weka.core.Randomizable;

import java.util.Arrays;
import java.util.HashSet;

public class ClassifierLists {
    // how to reference memebers in a hashset?
    public static String[] allUnivariate = new String[]{"DTW", "DTWCV", "ApproxElasticEnsemble", "ProximityForest", "ElasticEnsemble", "FastElasticEnsemble", "DD_DTW", "DTD_C", "NN_CID", "MSM", "TWE", "WDTW", "BOSS", "BOP", "SAXVSM", "SAX_1NN", "WEASEL", "cBOSS", "BOSSC45", "S-BOSS", "BoTSWEnsemble", "LPS", "TSF", "cTSF", "RISE", "FastShapelets", "LearnShapelets", "ShapeletTransformClassifier", "HiveCote", "FlatCote"};
    public static HashSet<String> allClassifiers;
    public static String[] distance;
    public static HashSet<String> distanceBased;
    public static String[] dictionary;
    public static HashSet<String> dictionaryBased;
    public static String[] interval;
    public static HashSet<String> intervalBased;
    public static String[] frequency;
    public static HashSet<String> frequencyBased;
    public static String[] shapelet;
    public static HashSet<String> shapeletBased;
    public static String[] hybrids;
    public static HashSet<String> hybridBased;
    public static String[] allMultivariate;
    public static HashSet<String> multivariateBased;
    public static String[] standard;
    public static HashSet<String> standardClassifiers;
    public static String[] bespoke;
    public static HashSet<String> bespokeClassifiers;

    public ClassifierLists() {
    }

    private static Classifier setDistanceBased(ExperimentalArguments exp) {
        String classifier = exp.classifierName;
        Classifier c = null;
        int fold = exp.foldId;
        byte var5 = -1;
        switch(classifier.hashCode()) {
            // how to add memebers to setDistanceBased and specify a case for them
            case -1986962017:
                if (classifier.equals("NN_CID")) {
                    var5 = 8;
                }
                break;
            case -1641428124:
                if (classifier.equals("ApproxElasticEnsemble")) {
                    var5 = 2;
                }
                break;
            case 2208:
                if (classifier.equals("EE")) {
                    var5 = 0;
                }
                break;
            case 75244:
                if (classifier.equals("LEE")) {
                    var5 = 1;
                }
                break;
            case 65370232:
                if (classifier.equals("DTD_C")) {
                    var5 = 6;
                }
                break;
            case 88186594:
                if (classifier.equals("FastElasticEnsemble")) {
                    var5 = 4;
                }
                break;
            case 623736156:
                if (classifier.equals("ProximityForest")) {
                    var5 = 3;
                }
                break;
            case 1488830118:
                if (classifier.equals("CID_DTW")) {
                    var5 = 7;
                }
                break;
            case 2012479880:
                if (classifier.equals("DD_DTW")) {
                    var5 = 5;
                }
        }

        switch(var5) {
            case 0:
                c = ElasticEnsemble.FACTORY.EE_V2.build();
                break;
            case 1:
                c = ElasticEnsemble.FACTORY.LEE.build();
                break;
            case 2:
                c = new ApproxElasticEnsemble();
                break;
            case 3:
                c = new ProximityForestWrapper();
                break;
            case 4:
                c = new FastElasticEnsemble();
                break;
            case 5:
                c = new DD_DTW();
                break;
            case 6:
                c = new DTD_C();
                break;
            case 7:
                c = new NN_CID();
                ((NN_CID)c).useDTW();
                break;
            case 8:
                c = new NN_CID();
                break;
            default:
                System.out.println("Unknown distance based classifier " + classifier + " should not be able to get here ");
                System.out.println("There is a mismatch between array distance and the switch statement ");
                throw new UnsupportedOperationException("Unknown distance based  classifier " + classifier + " should not be able to get here. There is a mismatch between array distance and the switch statement.");
        }

        return (Classifier)c;
    }

    private static Classifier setDictionaryBased(ExperimentalArguments exp) {
        String classifier = exp.classifierName;
        int fold = exp.foldId;
        byte var5 = -1;
        switch(classifier.hashCode()) {
            case -1875144537:
                if (classifier.equals("S-BOSS")) {
                    var5 = 7;
                }
                break;
            case -1856011994:
                if (classifier.equals("SAXVSM")) {
                    var5 = 1;
                }
                break;
            case -1738489817:
                if (classifier.equals("WEASEL")) {
                    var5 = 9;
                }
                break;
            case -1701561412:
                if (classifier.equals("SAX_1NN")) {
                    var5 = 2;
                }
                break;
            case -1002219951:
                if (classifier.equals("SpatialBOSS")) {
                    var5 = 6;
                }
                break;
            case -897016748:
                if (classifier.equals("BoTSWEnsemble")) {
                    var5 = 8;
                }
                break;
            case 65955:
                if (classifier.equals("BOP")) {
                    var5 = 0;
                }
                break;
            case 2044781:
                if (classifier.equals("BOSS")) {
                    var5 = 3;
                }
                break;
            case 93473360:
                if (classifier.equals("cBOSS")) {
                    var5 = 4;
                }
                break;
            case 786594679:
                if (classifier.equals("BOSSC45")) {
                    var5 = 5;
                }
        }

        Object c;
        switch(var5) {
            case 0:
                c = new BagOfPatterns();
                break;
            case 1:
                c = new SAXVSM();
                break;
            case 2:
                c = new SAXVSM();
                break;
            case 3:
                c = new BOSS();
                break;
            case 4:
                c = new cBOSS();
                break;
            case 5:
                c = new BOSSC45();
                break;
            case 6:
            case 7:
                c = new SpatialBOSS();
                break;
            case 8:
                c = new BoTSWEnsemble();
                break;
            case 9:
                c = new WEASEL();
                break;
            default:
                System.out.println("Unknown dictionary based classifier " + classifier + " should not be able to get here ");
                System.out.println("There is a mismatch between array dictionary and the switch statement ");
                throw new UnsupportedOperationException("Unknown dictionary based  classifier " + classifier + " should not be able to get here.There is a mismatch between array dictionary and the switch statement ");
        }

        return (Classifier)c;
    }

    private static Classifier setIntervalBased(ExperimentalArguments exp) {
        String classifier = exp.classifierName;
        int fold = exp.foldId;
        byte var5 = -1;
        switch(classifier.hashCode()) {
            case 75599:
                if (classifier.equals("LPS")) {
                    var5 = 0;
                }
                break;
            case 83367:
                if (classifier.equals("TSF")) {
                    var5 = 1;
                }
        }

        Object c;
        switch(var5) {
            case 0:
                c = new LPS();
                break;
            case 1:
                c = new TSF();
                break;
            default:
                System.out.println("Unknown interval based classifier " + classifier + " should not be able to get here ");
                System.out.println("There is a mismatch between array interval and the switch statement ");
                throw new UnsupportedOperationException("Unknown interval based  classifier " + classifier + " should not be able to get here.There is a mismatch between array interval and the switch statement ");
        }

        return (Classifier)c;
    }

    private static Classifier setFrequencyBased(ExperimentalArguments exp) {
        String classifier = exp.classifierName;
        int fold = exp.foldId;
        byte var5 = -1;
        switch(classifier.hashCode()) {
            case 2515657:
                if (classifier.equals("RISE")) {
                    var5 = 0;
                }
            default:
                switch(var5) {
                    case 0:
                        Classifier c = new RISE();
                        return c;
                    default:
                        System.out.println("Unknown interval based classifier, should not be able to get here ");
                        System.out.println("There is a mismatch between array interval and the switch statement ");
                        throw new UnsupportedOperationException("Unknown interval based  classifier, should not be able to get here There is a mismatch between array interval and the switch statement ");
                }
        }
    }

    private static Classifier setShapeletBased(ExperimentalArguments exp) {
        String classifier = exp.classifierName;
        int fold = exp.foldId;
        byte var5 = -1;
        switch(classifier.hashCode()) {
            case -873572419:
                if (classifier.equals("FastShapelets")) {
                    var5 = 1;
                }
                break;
            case 560391507:
                if (classifier.equals("ShapeletTransformClassifier")) {
                    var5 = 2;
                }
                break;
            case 1174901241:
                if (classifier.equals("ShapeletTreeClassifier")) {
                    var5 = 3;
                }
                break;
            case 1829151509:
                if (classifier.equals("LearnShapelets")) {
                    var5 = 0;
                }
        }

        Object c;
        switch(var5) {
            case 0:
                c = new LearnShapelets();
                break;
            case 1:
                c = new FastShapelets();
                break;
            case 2:
                c = new ShapeletTransformClassifier();
                break;
            case 3:
                c = new ShapeletTree();
                break;
            default:
                System.out.println("Unknown interval based classifier, should not be able to get here ");
                System.out.println("There is a mismatch between array interval and the switch statement ");
                throw new UnsupportedOperationException("Unknown interval based  classifier, should not be able to get here There is a mismatch between array interval and the switch statement ");
        }

        return (Classifier)c;
    }

    private static Classifier setHybridBased(ExperimentalArguments exp) {
        String classifier = exp.classifierName;
        int fold = exp.foldId;
        byte var5 = -1;
        switch(classifier.hashCode()) {
            case -1194446963:
                if (classifier.equals("HiveCote")) {
                    var5 = 1;
                }
                break;
            case -318789370:
                if (classifier.equals("TSCHIEF")) {
                    var5 = 2;
                }
                break;
            case 1690051350:
                if (classifier.equals("FlatCote")) {
                    var5 = 0;
                }
        }

        Object c;
        switch(var5) {
            case 0:
                c = new FlatCote();
                break;
            case 1:
                c = new HiveCote();
                ((HiveCote)c).setContract(48);
                break;
            case 2:
                c = new TSCHIEFWrapper();
                ((TSCHIEFWrapper)c).setSeed(fold);
                break;
            default:
                System.out.println("Unknown hybrid based classifier, should not be able to get here ");
                System.out.println("There is a mismatch between array hybrids and the switch statement ");
                throw new UnsupportedOperationException("Unknown hybrid based  classifier, should not be able to get here There is a mismatch between array hybrids and the switch statement ");
        }

        return (Classifier)c;
    }

    private static Classifier setMultivariate(ExperimentalArguments exp) {
        String classifier = exp.classifierName;
        int fold = exp.foldId;
        byte var5 = -1;
        switch(classifier.hashCode()) {
            case -1557939137:
                if (classifier.equals("Shapelet_D")) {
                    var5 = 1;
                }
                break;
            case -1557939132:
                if (classifier.equals("Shapelet_I")) {
                    var5 = 0;
                }
                break;
            case -1227391195:
                if (classifier.equals("Shapelet_Indep")) {
                    var5 = 2;
                }
                break;
            case 2123945:
                if (classifier.equals("ED_I")) {
                    var5 = 3;
                }
                break;
            case 65388489:
                if (classifier.equals("DTW_A")) {
                    var5 = 6;
                }
                break;
            case 65388492:
                if (classifier.equals("DTW_D")) {
                    var5 = 5;
                }
                break;
            case 65388497:
                if (classifier.equals("DTW_I")) {
                    var5 = 4;
                }
        }

        Object c;
        switch(var5) {
            case 0:
            case 1:
            case 2:
                c = new MultivariateShapeletTransformClassifier();
                ((MultivariateShapeletTransformClassifier)c).setOneDayLimit();
                ((MultivariateShapeletTransformClassifier)c).setSeed(fold);
                ((MultivariateShapeletTransformClassifier)c).setTransformType(classifier);
                break;
            case 3:
                c = new NN_ED_I();
                break;
            case 4:
                c = new NN_DTW_I();
                break;
            case 5:
                c = new NN_DTW_D();
                break;
            case 6:
                c = new NN_DTW_A();
                break;
            default:
                System.out.println("Unknown multivariate classifier, should not be able to get here ");
                System.out.println("There is a mismatch between multivariateBased and the switch statement ");
                throw new UnsupportedOperationException("Unknown multivariate classifier, should not be able to get here There is a mismatch between multivariateBased and the switch statement ");
        }

        return (Classifier)c;
    }

    private static Classifier setStandardClassifiers(ExperimentalArguments exp) {
        String classifier = exp.classifierName;
        int fold = exp.foldId;
        byte var5 = -1;
        switch(classifier.hashCode()) {
            case -1836950132:
                if (classifier.equals("SVMRBF")) {
                    var5 = 12;
                }
                break;
            case -1484183979:
                if (classifier.equals("BayesNet")) {
                    var5 = 6;
                }
                break;
            case -1407015436:
                if (classifier.equals("XGBoost")) {
                    var5 = 1;
                }
                break;
            case -996964626:
                if (classifier.equals("XGBoostMultiThreaded")) {
                    var5 = 0;
                }
                break;
            case 2124:
                if (classifier.equals("BN")) {
                    var5 = 13;
                }
                break;
            case 2207:
                if (classifier.equals("ED")) {
                    var5 = 7;
                }
                break;
            case 2484:
                if (classifier.equals("NB")) {
                    var5 = 9;
                }
                break;
            case 2496:
                if (classifier.equals("NN")) {
                    var5 = 17;
                }
                break;
            case 66052:
                if (classifier.equals("C45")) {
                    var5 = 8;
                }
                break;
            case 76433:
                if (classifier.equals("MLP")) {
                    var5 = 14;
                }
                break;
            case 2553199:
                if (classifier.equals("RotF")) {
                    var5 = 4;
                }
                break;
            case 2557762:
                if (classifier.equals("SVML")) {
                    var5 = 10;
                }
                break;
            case 2557767:
                if (classifier.equals("SVMQ")) {
                    var5 = 11;
                }
                break;
            case 63898478:
                if (classifier.equals("CAWPE")) {
                    var5 = 16;
                }
                break;
            case 78727329:
                if (classifier.equals("RandF")) {
                    var5 = 3;
                }
                break;
            case 301723398:
                if (classifier.equals("PLSNominalClassifier")) {
                    var5 = 5;
                }
                break;
            case 1889536911:
                if (classifier.equals("SmallTunedXGBoost")) {
                    var5 = 2;
                }
                break;
            case 2087573120:
                if (classifier.equals("Logistic")) {
                    var5 = 15;
                }
        }

        Object c;
        switch(var5) {
            case 0:
                c = new TunedXGBoost();
                break;
            case 1:
                c = new TunedXGBoost();
                ((TunedXGBoost)c).setRunSingleThreaded(true);
                break;
            case 2:
                c = new TunedXGBoost();
                ((TunedXGBoost)c).setRunSingleThreaded(true);
                TunedXGBoost var10000 = (TunedXGBoost)c;
                TunedXGBoost.setSmallParaSearchSpace_64paras();
                break;
            case 3:
                RandomForest r = new RandomForest();
                r.setNumIterations(500);
                c = r;
                break;
            case 4:
                RotationForest rf = new RotationForest();
                rf.setNumIterations(200);
                c = rf;
                break;
            case 5:
                c = new PLSNominalClassifier();
                break;
            case 6:
                c = new BayesNet();
                break;
            case 7:
                c = KNNLOOCV.FACTORY.ED_1NN_V1.build();
                break;
            case 8:
                c = new J48();
                break;
            case 9:
                c = new TunedClassifier();
                break;
            case 10:
                c = new NaiveBayes();
                break;
            case 11:
                c = new SMO();
                PolyKernel p = new PolyKernel();
                p.setExponent(1.0D);
                ((SMO)c).setKernel(p);
                ((SMO)c).setRandomSeed(fold);
                //((SMO)c).setBuildLogisticModels(true);
                break;
            case 12:
                c = new SMO();
                PolyKernel poly = new PolyKernel();
                poly.setExponent(2.0D);
                ((SMO)c).setKernel(poly);
                ((SMO)c).setRandomSeed(fold);
                //((SMO)c).setBuildLogisticModels(true);
                break;
            case 13:
                c = new SMO();
                RBFKernel rbf = new RBFKernel();
                rbf.setGamma(0.5D);
                ((SMO)c).setC(5.0D);
                ((SMO)c).setKernel(rbf);
                ((SMO)c).setRandomSeed(fold);
                //((SMO)c).setBuildLogisticModels(true);
                break;
            case 14:
                c = new BayesNet();
                break;
            case 15:
                c = new MultilayerPerceptron();
                break;
            case 16:
                c = new Logistic();
                break;
            case 17:
                c = new CAWPE();
                break;
            case 18:
                kNN k = new kNN(100);
                k.setCrossValidate(true);
                k.normalise(false);
                k.setDistanceFunction(new EuclideanDistance());
                return k;
            case 19:
                c = new RandomForest();
                break;
            default:
                System.out.println("Unknown standard classifier " + classifier + " should not be able to get here ");
                System.out.println("There is a mismatch between otherClassifiers and the switch statement ");
                throw new UnsupportedOperationException("Unknown standard classifier " + classifier + " should not be able to get here There is a mismatch between otherClassifiers and the switch statement ");
        }

        return (Classifier)c;
    }

    private static Classifier setBespokeClassifiers(ExperimentalArguments exp) {
        String classifier = exp.classifierName;
        String resultsPath = "";
        String dataset = "";
        int fold = exp.foldId;
        boolean canLoadFromFile = true;
        if (exp.resultsWriteLocation != null && exp.datasetName != null) {
            resultsPath = exp.resultsWriteLocation;
            dataset = exp.datasetName;
        } else {
            canLoadFromFile = false;
        }

        byte var8 = -1;
        switch(classifier.hashCode()) {
            case -1876617190:
                if (classifier.equals("HIVE-COTE")) {
                    var8 = 6;
                }
                break;
            case -1517544840:
                if (classifier.equals("HC-V2NoRise")) {
                    var8 = 0;
                }
                break;
            case -582551345:
                if (classifier.equals("HC-cSBOSS")) {
                    var8 = 4;
                }
                break;
            case 371302317:
                if (classifier.equals("HC-BcSBOSS")) {
                    var8 = 3;
                }
                break;
            case 457147446:
                if (classifier.equals("HIVE-COTEV2")) {
                    var8 = 1;
                }
                break;
            case 671940174:
                if (classifier.equals("TunedHIVE-COTE")) {
                    var8 = 7;
                }
                break;
            case 1438038065:
                if (classifier.equals("HC-TED2")) {
                    var8 = 2;
                }
                break;
            case 1954409304:
                if (classifier.equals("HIVE-COTE2")) {
                    var8 = 5;
                }
        }

        Object c;
        String[] cls;
        switch(var8) {
            case 0:
                if (!canLoadFromFile) {
                    throw new UnsupportedOperationException("ERROR: currently only loading from file for CAWPE and no results file path has been set. Call setClassifier with an ExperimentalArguments object exp with exp.resultsWriteLocation (contains component classifier results) and exp.datasetName set");
                }

                cls = new String[]{"CIF", "TED", "STC", "PF"};
                c = new HIVE_COTE();
                ((HIVE_COTE)c).setFillMissingDistsWithOneHotVectors(true);
                ((HIVE_COTE)c).setSeed(fold);
                ((HIVE_COTE)c).setBuildIndividualsFromResultsFiles(true);
                ((HIVE_COTE)c).setResultsFileLocationParameters(resultsPath, dataset, fold);
                ((HIVE_COTE)c).setClassifiersNamesForFileRead(cls);
                break;
            case 1:
                if (!canLoadFromFile) {
                    throw new UnsupportedOperationException("ERROR: currently only loading from file for CAWPE and no results file path has been set. Call setClassifier with an ExperimentalArguments object exp with exp.resultsWriteLocation (contains component classifier results) and exp.datasetName set");
                }

                cls = new String[]{"CIF", "TED", "RISE", "STC", "PF"};
                c = new HIVE_COTE();
                ((HIVE_COTE)c).setFillMissingDistsWithOneHotVectors(true);
                ((HIVE_COTE)c).setSeed(fold);
                ((HIVE_COTE)c).setBuildIndividualsFromResultsFiles(true);
                ((HIVE_COTE)c).setResultsFileLocationParameters(resultsPath, dataset, fold);
                ((HIVE_COTE)c).setClassifiersNamesForFileRead(cls);
                break;
            case 2:
                if (!canLoadFromFile) {
                    throw new UnsupportedOperationException("ERROR: currently only loading from file for CAWPE and no results file path has been set. Call setClassifier with an ExperimentalArguments object exp with exp.resultsWriteLocation (contains component classifier results) and exp.datasetName set");
                }

                cls = new String[]{"TSF", "BcS-BOSS", "RISE", "STC"};
                c = new HIVE_COTE();
                ((HIVE_COTE)c).setFillMissingDistsWithOneHotVectors(true);
                ((HIVE_COTE)c).setSeed(fold);
                ((HIVE_COTE)c).setBuildIndividualsFromResultsFiles(true);
                ((HIVE_COTE)c).setResultsFileLocationParameters(resultsPath, dataset, fold);
                ((HIVE_COTE)c).setClassifiersNamesForFileRead(cls);
                break;
            case 3:
                if (!canLoadFromFile) {
                    throw new UnsupportedOperationException("ERROR: currently only loading from file for CAWPE and no results file path has been set. Call setClassifier with an ExperimentalArguments object exp with exp.resultsWriteLocation (contains component classifier results) and exp.datasetName set");
                }

                cls = new String[]{"TSF", "BcS-BOSS", "RISE", "STC", "EE"};
                c = new HIVE_COTE();
                ((HIVE_COTE)c).setFillMissingDistsWithOneHotVectors(true);
                ((HIVE_COTE)c).setSeed(fold);
                ((HIVE_COTE)c).setBuildIndividualsFromResultsFiles(true);
                ((HIVE_COTE)c).setResultsFileLocationParameters(resultsPath, dataset, fold);
                ((HIVE_COTE)c).setClassifiersNamesForFileRead(cls);
                break;
            case 4:
                if (!canLoadFromFile) {
                    throw new UnsupportedOperationException("ERROR: currently only loading from file for CAWPE and no results file path has been set. Call setClassifier with an ExperimentalArguments object exp with exp.resultsWriteLocation (contains component classifier results) and exp.datasetName set");
                }

                cls = new String[]{"TSF", "cS-BOSS", "RISE", "STC", "EE"};
                c = new HIVE_COTE();
                ((HIVE_COTE)c).setFillMissingDistsWithOneHotVectors(true);
                ((HIVE_COTE)c).setSeed(fold);
                ((HIVE_COTE)c).setBuildIndividualsFromResultsFiles(true);
                ((HIVE_COTE)c).setResultsFileLocationParameters(resultsPath, dataset, fold);
                ((HIVE_COTE)c).setClassifiersNamesForFileRead(cls);
                break;
            case 5:
                if (!canLoadFromFile) {
                    throw new UnsupportedOperationException("ERROR: currently only loading from file for CAWPE and no results file path has been set. Call setClassifier with an ExperimentalArguments object exp with exp.resultsWriteLocation (contains component classifier results) and exp.datasetName set");
                }

                cls = new String[]{"TSF", "BOSS", "RISE", "STC"};
                c = new HIVE_COTE();
                ((HIVE_COTE)c).setFillMissingDistsWithOneHotVectors(true);
                ((HIVE_COTE)c).setSeed(fold);
                ((HIVE_COTE)c).setBuildIndividualsFromResultsFiles(true);
                ((HIVE_COTE)c).setResultsFileLocationParameters(resultsPath, dataset, fold);
                ((HIVE_COTE)c).setClassifiersNamesForFileRead(cls);
                break;
            case 6:
                if (!canLoadFromFile) {
                    throw new UnsupportedOperationException("ERROR: currently only loading from file for CAWPE and no results file path has been set. Call setClassifier with an ExperimentalArguments object exp with exp.resultsWriteLocation (contains component classifier results) and exp.datasetName set");
                }

                cls = new String[]{"TSF", "BOSS", "RISE", "STC", "EE"};
                c = new HIVE_COTE();
                ((HIVE_COTE)c).setFillMissingDistsWithOneHotVectors(true);
                ((HIVE_COTE)c).setSeed(fold);
                ((HIVE_COTE)c).setBuildIndividualsFromResultsFiles(true);
                ((HIVE_COTE)c).setResultsFileLocationParameters(resultsPath, dataset, fold);
                ((HIVE_COTE)c).setClassifiersNamesForFileRead(cls);
                break;
            case 7:
                if (!canLoadFromFile) {
                    throw new UnsupportedOperationException("ERROR: currently only loading from file for CAWPE and no results file path has been set. Call setClassifier with an ExperimentalArguments object exp with exp.resultsWriteLocation (contains component classifier results) and exp.datasetName set");
                }

                cls = new String[]{"TSF", "BOSS", "RISE", "STC", "EE"};
                HIVE_COTE hc = new HIVE_COTE();
                hc.setFillMissingDistsWithOneHotVectors(true);
                hc.setSeed(fold);
                hc.setBuildIndividualsFromResultsFiles(true);
                hc.setResultsFileLocationParameters(resultsPath, dataset, fold);
                hc.setClassifiersNamesForFileRead(cls);
                TunedClassifier tuner = new TunedClassifier();
                tuner.setClassifier(hc);
                ParameterSpace pc = new ParameterSpace();
                double[] alphaVals = new double[]{1.0D, 2.0D, 3.0D, 4.0D, 5.0D, 6.0D, 7.0D, 8.0D, 9.0D, 10.0D};
                pc.addParameter("a", alphaVals);
                tuner.setParameterSpace(pc);
                c = tuner;
                break;
            default:
                System.out.println("Unknown bespoke classifier, should not be able to get here ");
                System.out.println("There is a mismatch between bespokeClassifiers and the switch statement ");
                throw new UnsupportedOperationException("Unknown bespoke classifier, should not be able to get here There is a mismatch between bespokeClassifiers and the switch statement ");
        }

        return (Classifier)c;
    }

    public static Classifier setClassifier(ExperimentalArguments exp) {
        String classifier = exp.classifierName;
        Classifier c = null;
        if (distanceBased.contains(classifier)) {
            c = setDistanceBased(exp);
        } else if (dictionaryBased.contains(classifier)) {
            c = setDictionaryBased(exp);
        } else if (intervalBased.contains(classifier)) {
            c = setIntervalBased(exp);
        } else if (frequencyBased.contains(classifier)) {
            c = setFrequencyBased(exp);
        } else if (shapeletBased.contains(classifier)) {
            c = setShapeletBased(exp);
        } else if (hybridBased.contains(classifier)) {
            c = setHybridBased(exp);
        } else if (multivariateBased.contains(classifier)) {
            c = setMultivariate(exp);
        } else if (standardClassifiers.contains(classifier)) {
            c = setStandardClassifiers(exp);
        } else {
            if (!bespokeClassifiers.contains(classifier)) {
                System.out.println("Unknown classifier " + classifier + " it is not in any of the sublists ");
                throw new UnsupportedOperationException("Unknown classifier " + classifier + " it is not in any of the sublists on ClassifierLists ");
            }

            c = setBespokeClassifiers(exp);
        }

        if (c instanceof Randomizable) {
            ((Randomizable)c).setSeed(exp.foldId);
        }

        return c;
    }

    public static Classifier setClassifierClassic(String classifier, int fold) {
        ExperimentalArguments exp = new ExperimentalArguments();
        exp.classifierName = classifier;
        exp.foldId = fold;
        return setClassifier(exp);
    }

    public static void main(String[] args) throws Exception {
        System.out.println("Testing set classifier by running through the list in ClassifierLists.allUnivariate and ClassifierLists.allMultivariate");
        String[] var1 = allUnivariate;
        int var2 = var1.length;

        int var3;
        String str;
        Classifier c;
        for(var3 = 0; var3 < var2; ++var3) {
            str = var1[var3];
            System.out.println("Initialising " + str);
            c = setClassifierClassic(str, 0);
            System.out.println("Returned classifier " + c.getClass().getSimpleName());
        }

        var1 = allMultivariate;
        var2 = var1.length;

        for(var3 = 0; var3 < var2; ++var3) {
            str = var1[var3];
            System.out.println("Initialising " + str);
            c = setClassifierClassic(str, 0);
            System.out.println("Returned classifier " + c.getClass().getSimpleName());
        }

        var1 = standard;
        var2 = var1.length;

        for(var3 = 0; var3 < var2; ++var3) {
            str = var1[var3];
            System.out.println("Initialising " + str);
            c = setClassifierClassic(str, 0);
            System.out.println("Returned classifier " + c.getClass().getSimpleName());
        }

        var1 = bespoke;
        var2 = var1.length;

        for(var3 = 0; var3 < var2; ++var3) {
            str = var1[var3];
            System.out.println("Initialising " + str);
            c = setClassifierClassic(str, 0);
            System.out.println("Returned classifier " + c.getClass().getSimpleName());
        }

    }

    static {
        allClassifiers = new HashSet(Arrays.asList(allUnivariate));
        distance = new String[]{"DTW", "DTWCV", "ApproxElasticEnsemble", "ProximityForest", "FastElasticEnsemble", "DD_DTW", "DTD_C", "NN_CID", "EE", "LEE", "TUNED_DTW_1NN_V1"};
        distanceBased = new HashSet(Arrays.asList(distance));
        dictionary = new String[]{"BOSS", "BOP", "SAXVSM", "SAX_1NN", "WEASEL", "cBOSS", "BOSSC45", "S-BOSS", "SpatialBOSS", "BoTSWEnsemble"};
        dictionaryBased = new HashSet(Arrays.asList(dictionary));
        interval = new String[]{"LPS", "TSF", "cTSF"};
        intervalBased = new HashSet(Arrays.asList(interval));
        frequency = new String[]{"RISE"};
        frequencyBased = new HashSet(Arrays.asList(frequency));
        shapelet = new String[]{"FastShapelets", "LearnShapelets", "ShapeletTransformClassifier", "ShapeletTreeClassifier"};
        shapeletBased = new HashSet(Arrays.asList(shapelet));
        hybrids = new String[]{"HiveCote", "FlatCote", "TSCHIEF"};
        hybridBased = new HashSet(Arrays.asList(hybrids));
        allMultivariate = new String[]{"Shapelet_I", "Shapelet_D", "Shapelet_Indep", "ED_I", "DTW_I", "DTW_D", "DTW_A"};
        multivariateBased = new HashSet(Arrays.asList(allMultivariate));
        standard = new String[]{"XGBoostMultiThreaded", "XGBoost", "SmallTunedXGBoost", "RandF", "RotF", "PLSNominalClassifier", "BayesNet", "ED", "C45", "TunedC45", "SVML", "SVMQ", "SVMRBF", "MLP", "Logistic", "CAWPE", "NN", "RandF500"};
        standardClassifiers = new HashSet(Arrays.asList(standard));
        bespoke = new String[]{"HC-V2NoRise", "HIVE-COTEV2", "HIVE-COTE2", "HC-TED2", "HC-BcSBOSS", "HC-cSBOSS", "HIVE-COTE", "TunedHIVE-COTE"};
        bespokeClassifiers = new HashSet(Arrays.asList(bespoke));
    }
}

