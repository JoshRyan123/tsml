package Laboratories.lab3;

import experiments.data.DatasetLoading;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.j48.Distribution;
import weka.classifiers.trees.j48.EntropyBasedSplitCrit;
import weka.classifiers.trees.j48.InfoGainSplitCrit;
import weka.core.Instance;
import weka.core.Instances;

import java.io.IOException;
import java.text.DecimalFormat;

public class LiveClassDecisionTrees {

    public static void capabilitiesExample() throws Exception {
        // most importantly: what classifiers can handle which types of data

        // data can be categorical: labelled data such as colour or country
        // can be continuous: real valued
        // can be ordinal: discrete but ordered
        // can be strings, can be missing

        // how classifiers handle a data category (if they can) is highly influential on performance
        // some classifiers will handle some data categories less well than others
        // for example some decision trees build nodes on every discrete value, this can lead to ridiculously sized trees

        // both of the following use information gain
        J48 c45 = new J48(); //(c4.5)
        Id3 id3 = new Id3();

        System.out.println(" Base class capabilities = "+ new GetCapabilitiesExample().getCapabilities());

        System.out.println(" C45 capabilities = "+c45.getCapabilities());
        System.out.println(" ID3 capabilities = "+id3.getCapabilities());

        Instances iris = DatasetLoading.loadData("Data/lab1/Iris.arff");
        Instances playgolf = DatasetLoading.loadData("Data/lab1/PlayGolf.arff");
        // c45  for real valued attributes ; looks for binary splits
        // what it does it:
        // orders the values of an attribute
        // takes a cut point and works out info gain for that cut point
        // do this for all possible cut points
        // then finds the interval cutpoint which is best
        // f4 <= -0.78: 0 (50.0)                                   (branch1)(prediction is zero, which covers 50 cases)
        // f4 <= -0.78                                             (if > -0.78 proceed to next branch)
        //     f4 <= 0.65                                          (branch2)
        //         f3 <= 0.64: 1 (48.0/1.0)
        //         f3 <= 0.64
        //             f4 <= 0.39: 2 (3.0)
        //             f4 > 0.39: 1 (3.0/1.0)
        //     f4 > 0.65: 2 (46.0/1.0)

        c45.buildClassifier(iris);
        System.out.println("C45 Tree:"+c45);
        c45.buildClassifier(playgolf);
        System.out.println("C45 Tree:"+c45);
        // use with iris numeric values does not work
        // id3.buildClassifier(someData);
        id3.buildClassifier(playgolf);
        System.out.println("ID3 Tree:"+id3);
    }

    public static void  IGExample() throws IOException {
        Instances playGolf= DatasetLoading.loadData("Data/lab1/PlayGolf.arff");

        // Form a count matrix for outlook, contains the number of values for the attribute outlook, and number
        // of class values for the class variable
        int[][] outlook = new int[playGolf.attribute("Outlook").numValues()][playGolf.numClasses()];

        // loop for each variable and recover the value of the attribute and recover the class value, and increment by one
        for(Instance ins:playGolf){
            outlook[(int)ins.value(0)][(int)ins.classValue()]++;
        }
        for(int[] x:outlook) {
            for (int y : x)
                System.out.print(y + ",");
            System.out.print("\n");
        }

        //Play golf counts: Each row is an attribute value, each column a class value
        // give a count for the split on temperature, outlook, humidity, and windy

        // encoding of classes (2d array of counts representing a histogram)
        double[][] t= {{2,2},{7,3}};
        double[][] o= {{2,3},{0,4},{3,2}};
        double[][] h=  {{5,4},{4,1}};
        double[][] w=  {{6,2},{3,3}};

        DecimalFormat df = new DecimalFormat("##.###");

        // how to get information gain out of J48
        InfoGainSplitCrit infoGain = new InfoGainSplitCrit();

        //////////////////////////////////////////////////////////////////
        EntropyBasedSplitCrit entropyBased = new EntropyBasedSplitCrit() {
            @Override
            public String getRevision() {
                return null;
            }
        };
        //////////////////////////////////////////////////////////////////

        // stores the counts above and works out the probabilities (so we know the class probabilities; a proportion of
        // each class value within the overall data (thats our root node) then we look at the effect of splitting by
        // the attribute on those distributions to see whether it is good or not
        Distribution distO= new Distribution(o);

        System.out.println(" Dist for outlook= "+distO.dumpDistribution());

        // for outlook info gain (using IG and Dist)
        double outlookIG = infoGain.splitCritValue(distO);
        System.out.println(" Outlook IG = "+df.format(1/outlookIG));


        // temp

        Distribution distT= new Distribution(t);
        /*System.out.println(" Dist for tempurature= "+distT.dumpDistribution());*/
        double tempuratureIG = infoGain.splitCritValue(distT);
        System.out.println(" Tempurature IG = "+df.format(1/tempuratureIG));


        // humidity

        Distribution distH= new Distribution(h);
        /*System.out.println(" Dist for humidity= "+distH.dumpDistribution());*/
        double humidityIG = infoGain.splitCritValue(distH);
        System.out.println(" Humidity IG = "+df.format(1/humidityIG));


        // wind

        Distribution distW= new Distribution(w);
        /*System.out.println(" Dist for wind= "+distW.dumpDistribution());*/
        double windIG = infoGain.splitCritValue(distW);
        System.out.println(" Wind IG = "+df.format(1/windIG));
    }

    public static void playGolfExample(){

    }

    public static void main(String[] args) throws Exception {
        capabilitiesExample();
        IGExample();

    }

}
