package org.nornis.lightperceptron.utils;

public class Utils {

    /**
     * Calculates total square error
     *
     * @param  array of errors
     * @return total square error
     */
    public static double calcErrorSquare(double[] error) {
        double sum = 0;
        for (double e: error)
            sum += e * e;
        return sum;
    }
}
