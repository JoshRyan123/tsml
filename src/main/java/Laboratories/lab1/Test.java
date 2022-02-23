package Laboratories.lab1;

import weka.core.Instance;
import weka.core.Instances;

import java.io.FileReader;


public class Test {
    public static void main(String[] args) throws Exception {
        /** Instance – an object to store a single case/instance from a problem **/

        /** Instances – an object that stores a set of cases e.g. multiple
         Instance objects **/

        String firstDataLocation= "Data/lab1/Arsenal_TRAIN.arff";
        String secondDataLocation= "Data/lab1/Arsenal_TEST.arff";
//
//        Instances wdbc = DatasetLoading.loadData(firstDataLocation);
//
        /* Labsheet question 1 */
        Instances A = loadDataset(firstDataLocation);
        Instances B = loadDataset(secondDataLocation);
    }


    public static Instances loadDataset(String filename) throws Exception {
        int index = 0;
        try {
            // read file
            FileReader reader = new FileReader(filename);
            Instances D = new Instances(reader);

            // 1.2.print num attribute in test and instances in train
            if(filename == "Arsenal_TRAIN.arff") {
                System.out.println("\nDataset " + (String) filename + " Instances:\n" + D.numInstances());
            }
            if(filename == "Arsenal_TEST.arff") {
                System.out.println("\nDataset "+(String)filename+" Attributes:\n"+D.numAttributes());
            }

            // 3.Set class value. For a classification problem in Weka, we must explicitly set which attribute is
            //the class value
            D.setClassIndex(D.numAttributes() - 1);

            // 3.Print wins
            for(Instance inst:D){
                System.out.println("Class value "+(String)filename+" = " +inst.classValue());

                if(inst.classValue() == 2.0){
                    index++;
                }
            }
            System.out.println("Dataset Win count:\n"+index);

            //4.print fifth instance of test as double[]
            if(filename == "Arsenal_TEST.arff") {
                System.out.println("fifth instance values:");
                Instance first = D.instance(5);
                double[] firstArray = first.toDoubleArray();
                for(double i:firstArray){
                    System.out.println(i);
                }
                System.out.println("sixth instance values:");
                Instance second = D.instance(6);
                double[] secondArray = second.toDoubleArray();
                for(double i:secondArray){
                    System.out.println(i);
                }
            }


            //5.print out training instances using .toString method and remove saka
            if(filename == "Arsenal_TRAIN.arff") {
                for(Instance i: D){
                    toStringFormat(i);
                }
                // delete attribute saka from data and then check .toString again
                D.deleteAttributeAt(0);
                for(Instance i: D){
                    toStringFormat(i);
                }
            }

            //2.
            FileReader reader2 = new FileReader(filename);

            Instances C = new Instances(reader2);
            C.setClassIndex(C.numAttributes() - 1);

            MyClassifier cls = new MyClassifier();
            cls.buildClassifier(C);

            int count = 0;
            for(Instance i:C){
                //using model
                double predicted = cls.classifyInstance(i);
                //using actual result
                double actual = i.classValue();

                System.out.println(" Actual = "+actual+ " Predicted = "+predicted);
                if(predicted==actual)
                        count++;
            }
            System.out.println("Number correct:"+count);
            System.out.println("Accuracy:"+ count/(double)C.numInstances());

            count = 0;
            for(Instance i:C){
                double[] distribution = cls.distributionForInstance(i);
                System.out.println(" Distribution = "+distribution+ " index = "+count);
                count++;
                // Dont know why not printing correctly
                for(double j:distribution){
                    //individual distribution
                    System.out.println(j);
                }
            }

            return C;
        } catch(Exception e) {
            e.printStackTrace();
            throw new Exception("[Error] Failed to load Instances from file '"+filename+"'.");
        }
    }
    public static void toStringFormat(Instance instance){
        System.err.println("toString Instance\n"+ instance.toString());
    }
}


