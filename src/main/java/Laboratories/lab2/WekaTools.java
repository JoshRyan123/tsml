package Laboratories.lab2;

import Laboratories.lab1.MyClassifier;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileReader;

public class WekaTools {

    public void measureAccuracy(Instances data) throws Exception {
        MyClassifier cls = new MyClassifier();
        cls.buildClassifier(data);

        int count = 0;
        for (Instance i : data) {
            //using model
            double predicted = cls.classifyInstance(i);
            //using actual result
            double actual = i.classValue();

            System.out.println(" Actual = " + actual + " Predicted = " + predicted);
            if (predicted == actual)
                count++;
        }
        System.out.println("Number correct:" + count);
        System.out.println("Accuracy:" + count / (double) data.numInstances());
    }

    public static Instances loadData(String filename) throws Exception {
        try {
            Instances inst = null;
            FileReader reader = new FileReader(filename);
            inst = new Instances(reader);
            inst.setClassIndex(inst.numAttributes() - 1);
            reader.close();

            System.out.println("\nDataset " + (String) filename + " Instances:\n" + inst.numInstances());
            System.out.println("\nDataset "+ (String)filename+" Attributes:\n"+inst.numAttributes());

            return inst;
        } catch(Exception e) {
            e.printStackTrace();
            throw new Exception("[Error] Failed to load Instances from file '"+filename+"'.");
        }
    }

    public static void splitData(String path,String prob){
//        Instances all= DatasetLoading.loadDataNullable(path+prob+"\\"+prob);
//        Instances[] split= InstanceTools.resampleInstances(all, 0, 0.5);
//
//        OutFile out=new OutFile(path+prob+"\\"+prob+"_TRAIN.arff");
//        out.writeLine(split[0].toString());
//
//        out=new OutFile(path+prob+"\\"+prob+"_TEST.arff");
//        out.writeLine(split[1].toString());
    }

//    double[] classDistribution(Instances data){
//
//    }
}

