#ifndef KALMANFILTER_KALMANFILTER_H
#define KALMANFILTER_KALMANFILTER_H

#include "Eigen/Dense"

class KalmanFilter {
public:
    KalmanFilter() {
        ndim_ = 4;
        dt_ = 1.0;

        motion_mat_.setIdentity(2 * ndim_, 2 * ndim_);
        for (int i = 0; i < ndim_; ++i) {
            motion_mat_(i, ndim_ + i) = dt_;
        }
        update_mat_.setIdentity(ndim_, 2 * ndim_);

        std_weight_position_ = 1.0 / 20;
        std_weight_velocity_ = 1.0 / 160;
    }

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> initiate(const Eigen::VectorXd& measurement) {
        Eigen::VectorXd mean_pos = measurement;
        Eigen::VectorXd mean_vel = Eigen::VectorXd::Zero(mean_pos.size());
        Eigen::VectorXd mean(2 * ndim_);
        mean << mean_pos, mean_vel;

        Eigen::VectorXd std(8);
        std << 2 * std_weight_position_ * measurement(3),
                2 * std_weight_position_ * measurement(3),
                1e-2,
                2 * std_weight_position_ * measurement(3),
                10 * std_weight_velocity_ * measurement(3),
                10 * std_weight_velocity_ * measurement(3),
                1e-5,
                10 * std_weight_velocity_ * measurement(3);

        Eigen::MatrixXd covariance(2 * ndim_, 2 * ndim_);
        covariance.setZero();
        for (int i = 0; i < 2 * ndim_; ++i) {
            covariance(i, i) = std(i);
        }

        return std::make_pair(mean, covariance);
    }

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> predict(const Eigen::VectorXd& mean, const Eigen::MatrixXd& covariance) {
        Eigen::VectorXd std_pos(4);
        std_pos << std_weight_position_ * mean(2),
                std_weight_position_ * mean(3),
                1e-2,
                std_weight_position_ * mean(3);

        Eigen::VectorXd std_vel(4);
        std_vel << std_weight_velocity_ * mean(2),
                std_weight_velocity_ * mean(3),
                1e-5,
                std_weight_velocity_ * mean(3);

        Eigen::MatrixXd motion_cov(2 * ndim_, 2 * ndim_);
        motion_cov.setZero();
        for (int i = 0; i < 2 * ndim_; ++i) {
            motion_cov(i, i) = (i < ndim_) ? std_pos(i) : std_vel(i - ndim_);
        }

        Eigen::VectorXd new_mean = motion_mat_ * mean;
        Eigen::MatrixXd new_covariance = motion_mat_ * covariance * motion_mat_.transpose() + motion_cov;

        return std::make_pair(new_mean, new_covariance);
    }


    std::pair<Eigen::VectorXd, Eigen::MatrixXd> project(const Eigen::VectorXd& mean, const Eigen::MatrixXd& covariance) {
        Eigen::VectorXd std(4);
        std << std_weight_position_ * mean(3),
                std_weight_position_ * mean(3),
                1e-1,
                std_weight_position_ * mean(3);

        Eigen::MatrixXd innovation_cov(ndim_, ndim_);
        innovation_cov.setZero();
        for (int i = 0; i < ndim_; ++i) {
            innovation_cov(i, i) = std(i);
        }

        Eigen::VectorXd projected_mean = update_mat_ * mean;
        Eigen::MatrixXd projected_covariance = update_mat_ * covariance * update_mat_.transpose() + innovation_cov;

        return std::make_pair(projected_mean, projected_covariance);
    }

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> update(const Eigen::VectorXd& mean, const Eigen::MatrixXd& covariance, const Eigen::VectorXd& measurement) {
        auto projected = project(mean, covariance);
        Eigen::VectorXd projected_mean = projected.first;
        Eigen::MatrixXd projected_covariance = projected.second;

        Eigen::MatrixXd cholesky_factor = projected_covariance.llt().matrixL();
        Eigen::VectorXd innovation = measurement - projected_mean;
        Eigen::VectorXd z = cholesky_factor.triangularView<Eigen::Lower>().solve(innovation);

        Eigen::MatrixXd kalman_gain = covariance * update_mat_.transpose() * (update_mat_ * covariance * update_mat_.transpose() + projected_covariance).inverse();

        assert(kalman_gain.cols() == update_mat_.rows());

        Eigen::VectorXd new_mean = mean + kalman_gain * z;

        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(covariance.rows(), covariance.cols());
        Eigen::MatrixXd new_covariance = (I - kalman_gain * update_mat_) * covariance;

        return std::make_pair(new_mean, new_covariance);
    }


private:
    int ndim_;
    double dt_;
    Eigen::MatrixXd motion_mat_;
    Eigen::MatrixXd update_mat_;
    double std_weight_position_;
    double std_weight_velocity_;
};

#endif //KALMANFILTER_KALMANFILTER_H
