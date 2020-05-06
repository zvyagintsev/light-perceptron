package org.nornis.lightperceptron.activators;

public class Tanh implements IActivationFunction {

    @Override
    public double calculate(double net) {
        double exp2net = Math.exp(net);
        return (exp2net - 1) / (exp2net + 1);
    }

    @Override
    public double derivative(double tanhValue) {
        return 1 - tanhValue * tanhValue;
    }
}
