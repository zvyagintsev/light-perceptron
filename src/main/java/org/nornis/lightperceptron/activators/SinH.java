package org.nornis.lightperceptron.activators;

public class SinH implements IActivationFunction {
    @Override
    public double calculate(double net) {
        return (Math.exp(net) - Math.exp(-net)) / 2;
    }

    @Override
    public double derivative(double sinHValue) {
        return Math.sqrt(1 + sinHValue * sinHValue);
    }
}
