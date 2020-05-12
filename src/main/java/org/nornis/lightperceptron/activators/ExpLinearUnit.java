package org.nornis.lightperceptron.activators;

public class ExpLinearUnit implements IActivationFunction {
    @Override
    public double calculate(double net) {
        return net >= 0 ? (Math.exp(net) - 1) : net;
    }

    @Override
    public double derivative(double val) {
        return val < 0 ? 1 : val + 1;
    }
}
