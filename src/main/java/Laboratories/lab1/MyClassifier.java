package Laboratories.lab1;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class MyClassifier extends AbstractClassifier {
    double positiveMean;
    double negativeMean;
    //1.
    @Override
    public void buildClassifier(Instances data) throws Exception {
        positiveMean=negativeMean=0;
        int countP=0, countN=0;
        for(Instance ins:data){
            if(ins.classValue()==0){
                negativeMean+=ins.value(0);
                countN++;
            }
            else{
                positiveMean+=ins.value(0);
                countP++;
            }
            negativeMean/=countN;
            positiveMean/=countP;
        }

    }
    //2.
    // returns predicted class as double
    @Override
    public double classifyInstance(Instance data){
        /* If 1 star player is playing then predict Draw */
        /* 2 or 3 star players will predict a Win */
        /* No star players will predict a Lose */
        if(data.value(0) == 1)
            if(data.value(1) == 1)
                return 2.0;
            else if (data.value(2) == 1)
                return 2.0;
            else
                return 1.0;
        if(data.value(1) == 1)
            if(data.value(0) == 1)
                return 2.0;
            else if(data.value(2) == 1)
                return 2.0;
            else
                return 1.0;
        if(data.value(2) == 1)
            if(data.value(0) == 1)
                return 2.0;
            else if(data.value(1) == 1)
                return 2.0;
            else
                return 1.0;
        return 0.0;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] prob = new double[instance.numClasses()];
        double x = instance.value(0);
        double y = instance.value(1);
        double z = instance.value(2);
        if(x==1){
            prob[0] = 0.333;
        }
        if(y==1){
            prob[1] = 0.333;
        }
        if(z==1){
            prob[2] = 0.333;
        }
        return prob;
    }
}
