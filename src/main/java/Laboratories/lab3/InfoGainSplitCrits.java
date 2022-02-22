package Laboratories.lab3;

//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by FernFlower decompiler)
//

// IS THE IMPLEMENTATION FROM (    weka.classifiers.trees.j48.infoGainSplitCrit    )

import weka.classifiers.trees.j48.Distribution;
import weka.classifiers.trees.j48.EntropyBasedSplitCrit;
import weka.core.RevisionUtils;
import weka.core.Utils;

public final class InfoGainSplitCrits extends EntropyBasedSplitCrit {
    private static final long serialVersionUID = 4892105020180728499L;

    public InfoGainSplitCrits() {
    }

    // rather than maximising the gain the code instead minimizes the criteria
    public final double splitCritValue(Distribution bags) {
        double numerator = this.oldEnt(bags) - this.newEnt(bags);
        // we take the reciprocal value because we want to minimize the splitting criterion
        return Utils.eq(numerator, 0.0D) ? 1.7976931348623157E308D : bags.total() / numerator;
    }

    public final double splitCritValue(Distribution bags, double totalNoInst) {
        double noUnknown = totalNoInst - bags.total();
        double unknownRate = noUnknown / totalNoInst;
        double numerator = this.oldEnt(bags) - this.newEnt(bags);
        numerator = (1.0D - unknownRate) * numerator;
        return Utils.eq(numerator, 0.0D) ? 0.0D : numerator / bags.total();
    }

    public final double splitCritValue(Distribution bags, double totalNoInst, double oldEnt) {
        double noUnknown = totalNoInst - bags.total();
        double unknownRate = noUnknown / totalNoInst;
        double numerator = oldEnt - this.newEnt(bags);
        numerator = (1.0D - unknownRate) * numerator;
        return Utils.eq(numerator, 0.0D) ? 0.0D : numerator / bags.total();
    }

    public String getRevision() {
        return RevisionUtils.extract("$Revision: 10169 $");
    }
}