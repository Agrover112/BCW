
package org.deeplearning4j.examples.dataexamples;
import org.deeplearning4j.examples.dataexamples.BreastCancer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.io.IOException;

public class LoadModel {
    public static void main(String args[]) {
        try {
            BreastCancer.main(args);
            MultiLayerNetwork restored = MultiLayerNetwork.load(BreastCancer.locationToSave, true);

            System.out.println("Saved and loaded parameters are equal:      " + BreastCancer.model.params().equals(restored.params()));
             System.out.println("Saved and loaded configurations are equal:  " + BreastCancer.model.getLayerWiseConfigurations().equals(restored.getLayerWiseConfigurations()));
        }
        catch (Exception e) { e.getMessage();}
    }
}
