package org.nornis.lightperceptron;

import org.nornis.lightperceptron.activators.IActivationFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

public class Layer {

    private final static Logger logger = LoggerFactory.getLogger(Layer.class);

    // Layer weights
    protected double[][] weights;
    // Layer thresholds
    protected double[] thresholds;


    // number of input parameters
    final Integer nInput;
    // number of output parameters
    final Integer nOutput;

    public IActivationFunction getActivator() {
        return activator;
    }

    private final IActivationFunction activator;
    /**
     * The default constructor.
     * Weights and thresholds are generated randomly.
     *
     * @param nInput  number of input parameters
     * @param nOutput number of output parameters
     */
    public Layer(int nInput, int nOutput, IActivationFunction activationFunction) {
        //
        this.nInput = nInput;
        this.nOutput = nOutput;
        weights = new double[nOutput][nInput];
        thresholds = new double[nOutput];
        output = new double[nOutput];
        error = new double[nInput];
        activator = activationFunction;
        initWeights();
    }

    public Layer(double[][] weights, double[] thresholds, IActivationFunction activationFunction) {
        //
        this.nOutput = thresholds.length;
        this.nInput  = weights[0].length;
        this.weights = weights;
        this.thresholds = thresholds;
        output = new double[nOutput];
        error  = new double[nInput];
        activator = activationFunction;
    }


    public static final Double LEARN_RATE = 0.5;
    
    protected double[] output;
    protected double[] error;


    private void initWeights() {
        Random random = new Random();
        for (int i = 0; i < nOutput; i++)
            for (int j = 0; j < nInput; j++)
                weights[i][j] = random.nextDouble();
    }

    public void setWeights(double[][] weights) {
        // TODO: add validation
        this.weights = weights;
    }

    public double[][] getWeights() {
        return weights;
    }

    public void feedForward(double[] input) {

        for (int i = 0; i < nOutput; i++) {
            double sum = 0;
            for (int j = 0; j < nInput; j++)
                sum += weights[i][j] * input[j];
            sum += thresholds[i];
            output[i] = activator.calculate(sum);
        }
    }

    public double[] getOutput() {
        return output;
    }


    public void updateWeight(double[] input, double[] error, double learningRate) {
        for (int i = 0; i < nOutput; i++) {
            for (int j = 0; j < nInput; j++) {
                weights[i][j] += learningRate * error[i] * input[j];
            }
            thresholds[i] += learningRate * error[i];
        }
    }

    public double[] calcLayerError(
            final double[] forwardError,
            double[] input) {
        for (int i = 0; i < nInput; i++) {
            error[i] = 0;
            for (int j = 0; j < nOutput; j++)
                error[i] += forwardError[j] * weights[j][i];
            error[i] *= activator.derivative(input[i]);
        }
        return error;
    }

    public int getInputCount() {
        return this.nInput;
    }

    public int getOutputCount() {
        return this.nOutput;
    }
}
