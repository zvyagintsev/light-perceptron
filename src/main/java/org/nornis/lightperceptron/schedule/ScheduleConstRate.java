package org.nornis.lightperceptron.schedule;

public class ScheduleConstRate implements LearningSchedule {

    private final double LEARNING_RATE;

    public ScheduleConstRate(double learningRate) {
        LEARNING_RATE = learningRate;
    }


    @Override
    public double getRate(int stepInd) {
        return LEARNING_RATE;
    }
}
