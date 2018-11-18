package org.nornis.lightperceptron.activators;

public interface IActivationFunction {

    double calculate(double arg);
    double derivative(double arg);
}
