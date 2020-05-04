package org.nornis.lightperceptron;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import org.nornis.lightperceptron.activators.ActivationFunctionType;
import org.nornis.lightperceptron.activators.IActivationFunction;
import org.nornis.lightperceptron.schedule.LearningSchedule;
import org.nornis.lightperceptron.schedule.ScheduleConstRate;
import org.nornis.lightperceptron.utils.Constants;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class Perceptron implements IPerceptron {

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

    private final static double LEARNING_RATE = 0.2;
    private final static int LEARNING_STEPS = 100;

    private LearningSchedule schedule = null;

    @Override
    public void learning(double[][] data) {
        learning(data, LEARNING_STEPS);
    }

    @Override
    public void learning(double[][] data, int nSteps) {
        if (schedule == null) schedule = new ScheduleConstRate(LEARNING_RATE);
        for(int i = 0; i < nSteps; i++) {
            learning(data[i], i);
        }
    }

    @Override
    public double[] calculateOutput(double[] input) {
        double[] layerInput = new double[layers.get(0).nInput];
        System.arraycopy(input, 0, layerInput, 0, layerInput.length);
        for (Layer layer: layers) {
            layer.feedForward(layerInput, activationFunction);
            layerInput = layer.getOutput();
        }
        return layers.get(layers.size() - 1).getOutput();
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

    private void learning(double[] row, int iteration) {
        double[] output = calculateOutput(row);
        int nOutput = layers.get(layers.size() - 1).nOutput;
        // Calculating output error
        Layer ol = layers.get(layers.size() - 1);
        double[] forwardError = new double[nOutput];
        for (int i = 0; i < nOutput; i++) {
            // calculate layer error
            forwardError[i] = (row[nOutput + i] - output[i]) *
                    activationFunction.derivative(ol.getOutput()[i]);
        }

        for (int i = layers.size() - 1; i >= 0; i--) {
            Layer layer = layers.get(i);
            // previous layer error
            double[] layerInput = i == 0 ? row : layers.get(i - 1).getOutput();
            double[] err = layer.calcLayerError(forwardError, layerInput, activationFunction);
            layer.updateWeight(layerInput, forwardError,schedule.getRate(iteration));
            forwardError = err;
        };
    }

    public void setSchedule(LearningSchedule schedule) {
        this.schedule = schedule;
    }
}
