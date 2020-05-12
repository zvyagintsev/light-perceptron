package org.nornis.lightperceptron.schedule;

public class Schedule_1DivAi implements LearningSchedule {

    private final double A;

    public Schedule_1DivAi(double A) {
        this.A = A;
    }

    @Override
    public double getRate(int stepInd) {
        return 1.0 / (A * (stepInd + 1));
    }
}
