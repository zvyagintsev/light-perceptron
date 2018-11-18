package org.nornis.lightperceptron;

import org.junit.Test;

public class BuilderTest {

    @Test
    public void test1() {

        // Creating perceprton with hidden layer,
        // two input and 1 output
        Perceptron perceptron = new PerceptronBuilder().
                addLayer(2, 2).
                addLayer(2, 1).
                build();
        double result = 0;
        for (int i = 0; i < 100; i++) {
            perceptron.training(new double[]{0, 1}, new double[]{1});
            double currentResult = perceptron.calculate(new double[]{0, 1})[0];
            assert currentResult > result;
            result = currentResult;
        }
    }
}
