package org.nornis.lightperceptron.activators;

public class Sigmoid implements IActivationFunction {

    /**
     *  Returns value sigmoid function for NET = SUM (inpVal * weights) + thresholds
     *
     * @param net NET = SUM (inpVal * weights) + thresholds
     * @return neuron output
     */
    @Override
    public double calculate(double net) {
        return (1.0 / (1.0 + Math.exp(-net)));
    }

    /**
     *
     * @param sigmoidValue
     * @return
     */
    @Override
    public double derivative(double sigmoidValue) {
        return (sigmoidValue * (1.0 - sigmoidValue));
    }
}
