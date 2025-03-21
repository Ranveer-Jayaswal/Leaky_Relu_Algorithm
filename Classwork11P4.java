public class Classwork11P4 {
    public static void main(String[] args) {
        double[] w = {112, -123}; // weight is -2, 2
        double[] b = {-183, -182}; // bias is -6, -6
        int Num_Lines = 2;
        double[][] result = new double[Num_Lines][w.length];

        double[] x_train = {-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8};
        double[] y_train = {10, 8, 6, 4, 2, 0, 0, 0, 0, 0, 0, 0, 2, 4, 6, 8, 10};

        // y1 = -2x-6, y2= 2x+6
        result = GradDescend(w, b, x_train, y_train);
        System.out.println("w_1= " + result[0][0] + ", b_1= " + result[1][0]);
        System.out.println("w_2= " + result[0][1] + ", b_2= " + result[1][1]);
    }

    public static double Cost(double[] w, double[] b, double[] x_train, double[] y_train) {
        double totalCost = 0;
        double[] y_relu = new double[y_train.length];

        for (int i = 0; i < x_train.length; i++) {
            for (int j = 0; j < w.length; j++) {
                y_relu[i] += Relu(w[j] * x_train[i] + b[j]);
            }
            totalCost += Math.pow(y_relu[i] - y_train[i], 2);
        }
        return totalCost;
    }

    public static double Relu(double y) {
        return (y > 0) ? y : 0.001 * y; // Leaky Relu
    }

    public static double[][] GradDescend(double[] w, double[] b, double[] x_train, double[] y_train) {
        double[][] result = new double[2][w.length];
        double slope_w, intercept_b;
        double dw = 0.001;
        double db = 0.001;

        double[] w_estimate = w.clone();
        double[] w_estimate_up = new double[w.length];
        double[] w_estimate_down = new double[w.length];

        double[] b_estimate = b.clone();
        double[] b_estimate_up = new double[b.length];
        double[] b_estimate_down = new double[b.length];

        double dCost_w = 10, dCost_b = 10;
        double tolerance = 1E-9;
        int iteration = 0, resetCounter = 0, maxIterations = 10000000;

        double cost;

        while ((Math.abs(dCost_w) > tolerance || Math.abs(dCost_b) > tolerance) && iteration < maxIterations) {
            for (int i = 0; i < w.length; i++) {
                for (int j = 0; j < w.length; j++) {
                    if (i == j) {
                        w_estimate_up[j] = w_estimate[j] + dw;
                        w_estimate_down[j] = w_estimate[j] - dw;
                        b_estimate_up[j] = b_estimate[j] + db;
                        b_estimate_down[j] = b_estimate[j] - db;
                    } else {
                        w_estimate_up[j] = w_estimate[j];
                        w_estimate_down[j] = w_estimate[j];
                        b_estimate_up[j] = b_estimate[j];
                        b_estimate_down[j] = b_estimate[j];
                    }
                }

                cost = Cost(w_estimate, b_estimate, x_train, y_train);
                if (cost >= 10 && resetCounter >= 20) {
                    for (int m = 0; m < w.length; m++) {
                        w_estimate[m] = (int) (Math.random() * 200 - 100);
                        b_estimate[m] = (int) (Math.random() * 200 - 100);
                    }
                    resetCounter = 0;
                }

                dCost_w = Cost(w_estimate_up, b_estimate, x_train, y_train) - Cost(w_estimate_down, b_estimate, x_train, y_train);
                slope_w = dCost_w / (2 * dw);
                w_estimate[i] -= slope_w * dw;

                dCost_b = Cost(w_estimate, b_estimate_up, x_train, y_train) - Cost(w_estimate, b_estimate_down, x_train, y_train);
                intercept_b = dCost_b / (2 * db);
                b_estimate[i] -= intercept_b * db;

                if ((iteration % 1000000) == 0) {
                    cost = Cost(w_estimate, b_estimate, x_train, y_train);
                    System.out.println("Iteration= " + iteration + ", cost= " + cost + ", w_estimate= " + String.format("%.4f", w_estimate[0]) + ", b_estimate= " + String.format("%.4f", b_estimate[0]));
                    System.out.println(" ");
                }
            }

            resetCounter++;
            iteration++;

            if (iteration >= maxIterations) {
                System.out.println("w or b cannot converge");
            }
        }

        for (int i = 0; i < w.length; i++) {
            result[0][i] = w_estimate[i];
            result[1][i] = b_estimate[i];
        }
        return result;
    }
}
