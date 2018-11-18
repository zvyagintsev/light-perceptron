package org.nornis.lightperceptron.activators;

public enum ActivationFunctionType {
    SIGMOID;

    public static IActivationFunction getFunction(ActivationFunctionType type) {
        if(type == SIGMOID)
            return new Sigmoid();
        throw new IllegalArgumentException("Unknown function");
    }
}
