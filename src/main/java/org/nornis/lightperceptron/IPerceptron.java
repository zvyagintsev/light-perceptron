package org.nornis.lightperceptron;

import org.nornis.lightperceptron.schedule.LearningSchedule;

public interface IPerceptron {

    void learning(double[][] data);
    void learning(double[][] data, int nSteps);
    void learning(double[][] data, int nSteps, LearningSchedule learningSchedule);
    void learning(double[][] data, int nSteps, boolean withNoise);
    void learning(double[][] data, int nSteps, LearningSchedule learningSchedule, boolean withNoise);

    double[] calculateOutput(double[] input);

    String exportToJson();

    //double training(double[] input, double[] target);
    //double training(double[] input, double[] target, int iteration);

    //double training(double[][] input, double[][] target);
    //double training(double[][] input, double[][] target, int nCycles);
    //double training(double[][] input, double[][] target, int nCycles, boolean withRateScheduler);
    //double training(double[][] input, double[][] target, double error);

    //double training(double[][] data, int nInput, int nOutput);
    //double training(double[][] data, int nInput, int nOutput, int nCycles);
    //double training(double[][] data, int nInput, int nOutput, int nCycles, boolean withRateScheduler);
    //double training(double[][] data, int nInput, int nOutput, double error);

    //double[] calculate(double[] input);
    //double[] classification(double[] input);


}
