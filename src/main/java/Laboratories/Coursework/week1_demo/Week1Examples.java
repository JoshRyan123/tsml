package Laboratories.Coursework.week1_demo;

import experiments.data.DatasetLoading;
import weka.core.Instance;
import weka.core.Instances;

public class Week1Examples {
    public static void main(String[] args) throws Exception {
        // part 3
        Instances wdbc = DatasetLoading.loadData("src/main/java/ml6002b2022/week1_demo/wdbc");

        MyClassifier cls= new MyClassifier();
        cls.buildClassifier(wdbc);

        int count =0;
        for(Instance in:wdbc){
            double pred=cls.classifyInstance(in);
            double actual = in.classValue();

            System.out.println(" Actual = "+actual+" Predicted = "+pred);
            if(pred==actual) {
                count++;
            }

            double[] p = cls.distributionForInstance(in);
            for(double d:p) {
                System.out.println(d);
            }
        }
        System.out.println(" Number correct = "+count);
        System.out.println(" Accuracy = "+count/(double)wdbc.numInstances());

/*
        // part 1

        FileReader reader = new FileReader("src/main/java/experiments/data/uci/iris/iris.arff");
        Instances iris = new Instances(reader);
        iris.setClassIndex(iris.numAttributes()-1);

        System.out.println(" Number of class values "+iris.numClasses());
        System.out.println(" Num instances ="+wdbc.numInstances());
        System.out.println(" Num attributes ="+wdbc.numAttributes());

        for(Instance inst:wdbc){
            System.out.println(" Class value = "+inst.classValue());
        }

        Attribute att = iris.attribute(0);
        System.out.println(att);

        Instance first = wdbc.instance(0);
        System.out.println(first);

        double[] raw = first.toDoubleArray();
        for(double d:raw)
            System.out.println(d);
*/
    }


}
