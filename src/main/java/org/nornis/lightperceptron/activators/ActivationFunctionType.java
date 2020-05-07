package org.nornis.lightperceptron.activators;

public enum ActivationFunctionType {
    SIGMOID, TANH;

    public static IActivationFunction getFunction(ActivationFunctionType type) {
        if(type == SIGMOID)
            return new Sigmoid();
        if (type == TANH)
            return new Tanh();
        throw new IllegalArgumentException("Unknown function");
    }
}
