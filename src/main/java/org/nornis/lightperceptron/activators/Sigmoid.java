package org.nornis.lightperceptron.activators;

public class Sigmoid implements IActivationFunction {

    /**
     *
     * @param arg
     * @return
     */
    @Override
    public double calculate(double arg) {
        return (1.0 / (1.0 + Math.exp(-arg)));
    }

    /**
     *
     * @param arg
     * @return
     */
    @Override
    public double derivative(double arg) {
        return (arg * (1.0 - arg));
    }
}
