package org.nornis.lightperceptron;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.nornis.lightperceptron.activators.ActivationFunctionType;

import java.util.ArrayList;
import java.util.List;

public class PerceptronBuilder {

    private final static Gson gson = new GsonBuilder().create();

    private int inputCount = 1;
    private int outputCount = 1;
/*
    private LearningSchedule learningSchedule;

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

    public PerceptronBuilder setLearningSchedule(LearningSchedule schedule) {
        this.learningSchedule = schedule;
        return this;
    }

    public PerceptronBuilder addLayer(int nInput, int nOutput) {
        return addLayer(nInput, nOutput, activationFunctionType);
    }

    public PerceptronBuilder addLayer(int nInput, int nOutput, ActivationFunctionType afType) {
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
*/
    private static ActivationFunctionType DEFAULT_ACTIVATION_FUNCTION = ActivationFunctionType.SIGMOID;

    private List<Integer> neuronsInLayerCount;
    private List<ActivationFunctionType> activationFunctionTypeList;

    private PerceptronBuilder(int nInput) {
        this.inputCount = nInput;
        neuronsInLayerCount = new ArrayList<>();
        activationFunctionTypeList = new ArrayList<>();
    }

    public static PerceptronBuilder createPerceptron(int nInput) {
        return new PerceptronBuilder(nInput);
    }

    public PerceptronBuilder addLayer(int nNeurons) {
        outputCount = nNeurons;
        neuronsInLayerCount.add(nNeurons);
        activationFunctionTypeList.add(DEFAULT_ACTIVATION_FUNCTION);
        return this;
    }

    public PerceptronBuilder addLayer(int nNeurons, ActivationFunctionType activationFunctionType) {
        outputCount = nNeurons;
        neuronsInLayerCount.add(nNeurons);
        activationFunctionTypeList.add(activationFunctionType);
        return this;
    }

    public Perceptron build() {
        List<Layer> layers = new ArrayList<>(neuronsInLayerCount.size());
        layers.add(new Layer(
                inputCount, neuronsInLayerCount.get(0),
                ActivationFunctionType.getFunction(activationFunctionTypeList.get(0)))
        );
        for(int i = 1; i < neuronsInLayerCount.size(); i++) {
            Layer layer = new Layer(
                    neuronsInLayerCount.get(i - 1),
                    neuronsInLayerCount.get(i),
                    ActivationFunctionType.getFunction(activationFunctionTypeList.get(i)));
            layers.add(layer);
        }
        return new Perceptron(layers);
    }

    /*
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
    }*/
}
