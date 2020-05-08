package org.nornis.lightperceptron;

import org.junit.Test;
import org.nornis.lightperceptron.activators.Sigmoid;

public class LayerTest {
    @Test
    public void forwardTest() {
        double[][] weights = {{1, 0.5}, {-1, 2}};
        double[] thresholds = {1, 1};

        double[] input = {0, 1};

        //Layer layer  = new Layer(weights, thresholds);

        //layer.feedForward(input, new Sigmoid());
        //double[] output = layer.getOutput();
        //assert Math.abs(output[0] - 0.81757) < 0.0001;
        //assert Math.abs(output[1] - 0.952574) < 0.0001;
    }
}
