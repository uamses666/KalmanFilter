#include "KalmanFilter.h"
#include <iostream>

int main() {
    KalmanFilter kf;

    Eigen::VectorXd measurement1(4);
    measurement1 << 1.0, 2.0, 0.5, 0.2;
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> init_result1 = kf.initiate(measurement1);
    std::cout << "Initiate Mean (Test 1):\n" << init_result1.first << std::endl;
    std::cout << "Initiate Covariance (Test 1):\n" << init_result1.second << std::endl;

    Eigen::VectorXd mean2(8);
    mean2 << 1.0, 2.0, 0.5, 0.2, 0.1, 0.2, 0.05, 0.02;
    Eigen::MatrixXd covariance2(8, 8);
    covariance2.setIdentity();
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> predict_result2 = kf.predict(mean2, covariance2);
    std::cout << "Predicted Mean (Test 2):\n" << predict_result2.first << std::endl;
    std::cout << "Predicted Covariance (Test 2):\n" << predict_result2.second << std::endl;

    Eigen::VectorXd mean3(8);
    mean3 << 1.0, 2.0, 0.5, 0.2, 0.1, 0.2, 0.05, 0.02;
    Eigen::MatrixXd covariance3(8, 8);
    covariance3.setIdentity();
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> project_result3 = kf.project(mean3, covariance3);
    std::cout << "Projected Mean (Test 3):\n" << project_result3.first << std::endl;
    std::cout << "Projected Covariance (Test 3):\n" << project_result3.second << std::endl;

    Eigen::VectorXd measurement_update4(4);
    measurement_update4 << 1.2, 2.2, 0.45, 0.18;
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> update_result4 = kf.update(mean3, covariance3, measurement_update4);
    std::cout << "Updated Mean (Test 4):\n" << update_result4.first << std::endl;
    std::cout << "Updated Covariance (Test 4):\n" << update_result4.second << std::endl;

    Eigen::VectorXd measurement5(4);
    measurement5 << 0.5, 1.0, 0.2, 0.1;
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> init_result5 = kf.initiate(measurement5);
    std::cout << "Initiate Mean (Test 5):\n" << init_result5.first << std::endl;
    std::cout << "Initiate Covariance (Test 5):\n" << init_result5.second << std::endl;

    return 0;
}

