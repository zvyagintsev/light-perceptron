package org.nornis.lightperceptron;

import org.junit.Test;

import java.util.Random;

public class TrainingTest {
    private final double[][] DATA_ANIMALS = {
            {2, 6, 0, 1, 0, 0, 0, 0}, // insect
            {2, 2, 1, 0, 1, 0, 0, 0}, // bird
            {0, 0, 1, 0, 0, 1, 0, 0}, // snake
            {0, 4, 0, 0, 0, 0, 1, 0}, // mammal
            {0, 8, 0, 0, 0, 0, 0, 1}  // spider

    };

    @Test
    public void animalsClassification1() {
        PerceptronBuilder pb = new PerceptronBuilder();
        Perceptron p = pb.
                addLayer(3, 10).
                addLayer(10, 10).
                addLayer(10, 5).
                build();

        p.training(DATA_ANIMALS, 3, 5, 10000);
        double[] result = p.classification(new double[] {0, 4, 1});
        assert result[3] != 0;

        result = p.classification(new double[] {2, 6, 0});
        assert result[0] != 0;
    }

    @Test
    public void approximation() {
        int maxIteration = 1000;
        PerceptronBuilder pb = new PerceptronBuilder();
        Perceptron p = pb.
                addLayer(2, 2).
                addLayer(2, 1).
                build();

        Random r = new Random();
        double a, b;

        for (int i = 0; i < maxIteration; i++) {
            // generate a, b, 0 <= a <= 1, 0 <= a <=
            a = r.nextDouble();
            b = r.nextDouble();
            p.training(
                    new double[]{a, b}, new double[]{Math.sin(a) * Math.cos(b)});
        }
        a = r.nextDouble();
        b = r.nextDouble();
        double[] calcResult = p.calculate(new double[] {a, b});
    }
}
