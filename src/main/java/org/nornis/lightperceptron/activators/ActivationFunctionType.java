package org.nornis.lightperceptron.activators;

public enum ActivationFunctionType {
    SIGMOID, TAN_H, SIN_H, ELU;

    public static IActivationFunction getFunction(ActivationFunctionType type) {
        if(type == SIGMOID)
            return new Sigmoid();
        else if (type == TAN_H)
            return new TanH();
        else if (type == SIN_H)
            return new SinH();
        else if (type == ELU)
            return new ExpLinearUnit();
        else throw new IllegalArgumentException("Unknown activation function");
    }
}
