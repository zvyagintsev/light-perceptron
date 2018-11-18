package org.nornis.lightperceptron;

import org.junit.Test;

import java.util.Arrays;

public class ImportExportTest {


    @Test
    public void animals() {

        final double[][] DATA_ANIMALS = {
                {2, 6, 0, 1, 0, 0, 0}, // insect
                {2, 2, 0, 0, 1, 0, 0}, // bird
                {0, 0, 1, 0, 0, 1, 0}, // snake
                {0, 4, 1, 0, 0, 0, 1}, // mammal
        };

        PerceptronBuilder pb = new PerceptronBuilder();
        Perceptron p = pb.
                addLayer(3, 5).
                //addLayer(10, 10).
                addLayer(5, 4).
                build();



        p.training(DATA_ANIMALS, 3, 4, 1000);
        double[] result = p.classification(new double[] {2, 6, 0});
        String str = p.exportToJson();

        Perceptron p2 = pb.loadFromJson(str);
        double[] result2 = p2.classification(new double[] {2, 6, 0});
        assert Arrays.equals(result, result2);
    }

}
