package org.nornis.lightperceptron;

import com.google.gson.*;
import org.nornis.lightperceptron.activators.ActivationFunctionType;
import org.nornis.lightperceptron.utils.Constants;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class PerceptronBuilder {

    private final static Gson gson = new GsonBuilder().create();

    private int inputCount = 1;
    private int outputCount = 1;

    // By default Sigmoid
    private ActivationFunctionType activationFunctionType = ActivationFunctionType.SIGMOID;

    private List<Layer> layers = new ArrayList<>();

    public PerceptronBuilder setInput(int nInput) {
        this.inputCount = nInput;
        return this;
    }

    public Perceptron build() {
        return new Perceptron(layers,
                ActivationFunctionType.getFunction(activationFunctionType));
    }

    public PerceptronBuilder setActivationFunction(ActivationFunctionType fType) {
        this.activationFunctionType = fType;
        return this;
    }

    public PerceptronBuilder addLayer(int nInput, int nOutput) {
        Layer layer = new Layer(nInput, nOutput);
        this.outputCount = layer.getOutputCount();
        layers.add(layer);
        return this;
    }

    public PerceptronBuilder addLayer(double[][] weights, double[] thresholds) {
        Layer layer = new Layer(weights, thresholds);
        this.outputCount = layer.getOutputCount();
        layers.add(layer);
        return this;
    }

    public Perceptron loadFromJson(String jsonStr) {
        JsonObject json = gson.fromJson(jsonStr, JsonObject.class);
        return loadFromJson(json);
    }

    public Perceptron loadFromJson(JsonObject json) {
        JsonArray layersArr = Optional.ofNullable(json.get(Constants.LAYERS_ARRAY)).
                map(JsonElement::getAsJsonArray).get();
        List<Layer> layers = new ArrayList<>(layersArr.size());
        layersArr.forEach((JsonElement layerElement) -> {
            JsonArray neuronsArray = layerElement.getAsJsonObject().
                    get(Constants.NEURONS_ARR).getAsJsonArray();
            double[] thresholds = new double[neuronsArray.size()];
            double[][] weights = new double[neuronsArray.size()][];

            for (int i = 0; i < neuronsArray.size(); i++) {
                JsonObject neuronElement = neuronsArray.get(i).getAsJsonObject();
                double threshold = neuronElement.getAsJsonObject().
                        get(Constants.THRESHOLD).getAsDouble();
                thresholds[i] = threshold;
                JsonArray weightsArray = neuronElement.getAsJsonObject().
                        get(Constants.WEIGHTS_ARRAY).getAsJsonArray();
                double[] neuronWeights = new double[weightsArray.size()];
                for(int j = 0; j < weightsArray.size(); j++) {
                    neuronWeights[j] = weightsArray.get(j).getAsDouble();
                }
                weights[i] = neuronWeights;
            }
            Layer layer = new Layer(weights, thresholds);
            layers.add(layer);
        });
        return new Perceptron(layers,
                ActivationFunctionType.getFunction(activationFunctionType));
    }
}
