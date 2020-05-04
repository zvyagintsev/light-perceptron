package org.nornis.lightperceptron.schedule;

public interface LearningSchedule {
    double getRate(int stepInd);
}
