package org.nornis.lightperceptron;

public interface INeuralNetwork {

    double training(double[] input, double[] target);

    double training(double[][] input, double[][] target);
    double training(double[][] input, double[][] target, int nCycles);
    //double training(double[][] input, double[][] target, double error);

    double training(double[][] data, int nInput, int nOutput);
    double training(double[][] data, int nInput, int nOutput, int nCycles);
    //double training(double[][] data, int nInput, int nOutput, double error);

    double[] calculate(double[] input);
    double[] classification(double[] input);

    String exportToJson();
}
