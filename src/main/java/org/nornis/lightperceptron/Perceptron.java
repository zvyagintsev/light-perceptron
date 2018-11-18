package org.nornis.lightperceptron;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import org.nornis.lightperceptron.activators.ActivationFunctionType;
import org.nornis.lightperceptron.activators.IActivationFunction;
import org.nornis.lightperceptron.utils.Constants;
import org.nornis.lightperceptron.utils.Utils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;

public class Perceptron implements INeuralNetwork {

    private final static Logger logger = LoggerFactory.getLogger(Perceptron.class);

    private List<Layer> layers;

    private int nInput;
    private int nOutput;

    IActivationFunction activationFunction;

    private ActivationFunctionType type;

    public Perceptron(List<Layer> layers, IActivationFunction aFunction) {
        this.layers = layers;
        this.activationFunction = aFunction;
        this.nInput  = layers.get(0).getInputCount();
        this.nOutput = layers.get(layers.size() - 1).getOutputCount();
    }

    /**
     *
     * @param target
     */
    @Override
    public double training(double[] input, double[] target) {
        double[] output = calculate(input);
        // Calculating output error
        Layer ol = layers.get(layers.size() - 1);
        double[] forwardError = new double[nOutput];
        for (int i = 0; i < nOutput; i++) {
            // calculate layer error
            forwardError[i] = (target[i] - output[i]) *
                    activationFunction.derivative(ol.getOutput()[i]);
        }

        for (int i = layers.size() - 1; i >= 0; i--) {
            Layer layer = layers.get(i);
            // previous layer error
            double[] layerInput = i == 0 ? input : layers.get(i - 1).getOutput();
            double[] err = layer.calcLayerError(forwardError, layerInput, activationFunction);

            layer.updateWeight(layerInput, forwardError);
            forwardError = err;
        }
        return Utils.calcErrorSquare(forwardError);
    }

    @Override
    public double training(double[][] input, double[][] target) {
        return training(input, target, 1);
    }

    @Override
    public double training(double[][] input, double[][] target, int nCycles) {
        if (input.length != target.length)
            throw new IllegalArgumentException("Input and target arrays have different numbers of rows.");
        if (nCycles < 1)
            throw new IllegalArgumentException("The number of training cycles can not be less than 0");

        double squireError = 0;
        for (int i = 0; i < nCycles; i++) {
            for (int j = 0; j < input.length; j++)
                squireError = training(input[j], target[j]);
        }
        return squireError;
    }

    /*@Override
    public double training(double[][] input, double[][] target, double maxError) {
        double squireError = 0;
        double nCycles = 0;
        do {
            for (int j = 0; j < input.length; j++)
                squireError = training(input[j], target[j]);
            nCycles++;
        } while (squireError > maxError ||
                        nCycles < Constants.MAX_ITERATION_COUNT);
        return squireError;
    }*/

    @Override
    public double training(double[][] data, int nInput, int nOutput) {
        double[][] input = new double[data.length][nInput];
        double[][] output = new double[data.length][nOutput];
        return training(input, output);
    }

    @Override
    public double training(double[][] data, int nInput, int nOutput, int nCycles) {
        double[][] input = new double[data.length][nInput];
        double[][] output = new double[data.length][nOutput];
        for(int i = 0; i < data.length; i++) {
            System.arraycopy(data[i], 0, input[i], 0, nInput);
            System.arraycopy(data[i], nInput, output[i], 0, nOutput);
        }
        return training(input, output, nCycles);
    }

    /*@Override
    public double training(double[][] data, int nInput, int nOutput, double error) {
        double[][] input = new double[data.length][nInput];
        double[][] output = new double[data.length][nOutput];
        return training(input, output, error);
    }*/

    /**
     * Calculates neural net output
     *
     * @param input
     * @return
     */
    @Override
    public double[] calculate(double[] input) {
        double[] layerInput = input;
        for (Layer layer: layers) {
            layer.feedForward(layerInput, activationFunction);
            layerInput = layer.getOutput();
        }
        return layers.get(layers.size() - 1).getOutput();
    }

    @Override
    public double[] classification(double[] input) {
        double[] calc = calculate(input);
        double[] result = new double[calc.length];

        double maxItem = calc[0];
        int index = 0;

        for(int i = 1; i < calc.length; i++) {
            if(calc[i] > maxItem) {
                maxItem = calc[i];
                index = i;
            }
        }
        Arrays.fill(result, 0);
        result[index] = 1;
        return result;
    }

    @Override
    public String exportToJson() {
        JsonObject nnJson = new JsonObject();
        JsonArray layersArray = new JsonArray();
        for (Layer layer: layers) {
            JsonObject layerElement = new JsonObject();
            layerElement.addProperty(Constants.INPUT_COUNT, layer.getInputCount());
            layerElement.addProperty(Constants.OUTPUT_COUNT, layer.getOutputCount());

            JsonArray neuronsArray = new JsonArray();

            for (int i = 0; i < layer.getOutputCount(); i++) {
                JsonObject neuronElement = new JsonObject();
                JsonArray weightsArray = new JsonArray();
                for (double item: layer.weights[i]) {
                    weightsArray.add(item);
                }
                neuronElement.add(Constants.WEIGHTS_ARRAY, weightsArray);
                neuronElement.addProperty(Constants.THRESHOLD, layer.thresholds[i]);

                neuronsArray.add(neuronElement);
            }
            layerElement.add(Constants.NEURONS_ARR, neuronsArray);
            layersArray.add(layerElement);
        }
        nnJson.add(Constants.LAYERS_ARRAY, layersArray);
        return nnJson.toString();
    }
}
