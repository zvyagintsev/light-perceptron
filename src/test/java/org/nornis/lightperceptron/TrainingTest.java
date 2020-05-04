package org.nornis.lightperceptron;

import org.junit.Test;

import java.util.Random;

public class TrainingTest {

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

        double [][] data = new double[maxIteration][];
        for (int i = 0; i < maxIteration; i++) {
            // generate a, b, 0 <= a <= 1, 0 <= a <= 1
            a = r.nextDouble();
            b = r.nextDouble();
            data[i] = new double[]{a, b, Math.sin(a) * Math.cos(b)};
        }
        p.learning(data, 1000);
        a = r.nextDouble();
        b = r.nextDouble();
        double[] calcResult = p.calculateOutput(new double[] {a, b});
    }
}
