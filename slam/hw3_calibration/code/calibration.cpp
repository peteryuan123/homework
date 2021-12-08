
#include <iostream>
#include <fstream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>

typedef Eigen::Vector3d point2d;
typedef Eigen::Vector4d point3d;
typedef std::pair<point2d, point3d> match;

int point_num = 8;
std::string folder_name = "../../material/CS284_hw3_data/yrz/";

Eigen::Matrix3d calculateR(Eigen::Matrix3d camera_matrix, int type)
{
    double c, s;
    Eigen::Matrix3d result;
    if (type == 0)
    {
        c = -camera_matrix(2,2) / sqrt(pow(camera_matrix(2,1) , 2) + pow(camera_matrix(2,2) , 2));
        s = camera_matrix(2,1) / sqrt(pow(camera_matrix(2,1) , 2) + pow(camera_matrix(2,2) , 2));
        result << 1, 0, 0,
                  0, c, -s,
                  0, s, c;
    }
    else if (type == 1)
    {
        c = camera_matrix(2,2) / sqrt(pow(camera_matrix(2,0) , 2) + pow(camera_matrix(2,2) , 2));
        s = camera_matrix(2,0) / sqrt(pow(camera_matrix(2,0) , 2) + pow(camera_matrix(2,2) , 2));
        result << c, 0, s,
                  0, 1, 0,
                 -s, 0, c;
    }
    else if (type == 2)
    {
        c = camera_matrix(1,1) / sqrt(pow(camera_matrix(1,0) , 2) + pow(camera_matrix(1,1) , 2));
        s = -camera_matrix(1,0) / sqrt(pow(camera_matrix(1,0) , 2) + pow(camera_matrix(1,1) , 2));
        result << c, -s, 0,
                s, c, 0,
                0, 0, 1;
    }

    return result;
}

Eigen::Matrix3d computeEssentialMatrix(std::vector<point2d>& pts0, std::vector<point2d>& pts1, Eigen::Matrix3d K0, Eigen::Matrix3d K1)
{
    Eigen::Matrix<double, Eigen::Dynamic, 9> coefficient_matrix;
    coefficient_matrix.resize(pts0.size(), 9);
    coefficient_matrix.setZero();

    for (int i = 0; i < pts0.size(); i++)
    {
        point2d pt0_normalized = K0.inverse() * pts0[i];
        point2d pt1_normalized = K1.inverse() * pts1[i];

        double x1 = pt0_normalized.x(), y1 = pt0_normalized.y();
        double x2 = pt1_normalized.x(), y2 = pt1_normalized.y();
        coefficient_matrix.block<1, 9>(i, 0) << x2*x1 , x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1;
    }
//    std::cout << coefficient_matrix << std::endl;

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(coefficient_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d essentialMatrix;
    essentialMatrix << svd.matrixV()(0, 8), svd.matrixV()(1, 8), svd.matrixV()(2, 8),
                       svd.matrixV()(3, 8), svd.matrixV()(4, 8), svd.matrixV()(5, 8),
                       svd.matrixV()(6, 8), svd.matrixV()(7, 8), svd.matrixV()(8, 8);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd2(essentialMatrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
//    std::cout << svd2.singularValues() << std::endl;
    Eigen::Vector3d correct_singular(1,1,0);
    essentialMatrix = svd2.matrixU() * correct_singular.asDiagonal() * svd2.matrixV().transpose();
    std::cout << "essentialMatrix" << std::endl;
    std::cout << essentialMatrix << std::endl;
    return essentialMatrix;
}

Eigen::Vector3d triangulate(point2d pt0, point2d pt1, Eigen::Matrix3d R, Eigen::Vector3d t, Eigen::Matrix3d &K0, Eigen::Matrix3d &K1)
{
    Eigen::Matrix<double,3,4> T1, T12;
    T1 <<   1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0;

    T12.block<3,3>(0,0) = R;
    T12.block<3,1>(0,3) = t;

    Eigen::Matrix<double,3,4> proj0, proj1;

    proj0 = K0 * T1;
    proj1 = K1 * T12;

    Eigen::Matrix4d A;
    A.row(0) = proj0.row(2)*pt0.x() - proj0.row(0);
    A.row(1) = proj0.row(2)*pt0.y() - proj0.row(1);
    A.row(2) = proj1.row(2)*pt1.x() - proj1.row(0);
    A.row(3) = proj1.row(2)*pt1.y() - proj1.row(1);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Vector4d p_w = svd.matrixV().col(3);
    p_w /= p_w(3,0);

    return Eigen::Vector3d(p_w(0),p_w(1),p_w(2));
}


Eigen::Matrix4d decomposeEssentialMatrix(Eigen::Matrix3d essential_matrix, std::vector<point2d>& pts0, std::vector<point2d>& pts1,
                              Eigen::Matrix3d K0, Eigen::Matrix3d K1)
{
    Eigen::Matrix3d w,z;
    w << 0, -1, 0,
         1, 0, 0,
         0, 0, 1;
    z << 0, 1, 0,
         -1, 0, 0,
         0, 0, 0;

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(essential_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::Matrix3d t_skew = svd.matrixU() * z * svd.matrixU().transpose();
    Eigen::Vector3d ta(-t_skew(1, 2), t_skew(0, 2), -t_skew(0, 1));
    Eigen::Vector3d tb = -ta;

    Eigen::Matrix3d Ra = svd.matrixU() * w * svd.matrixV().transpose();
    if (Ra.determinant() < 0) Ra = -Ra;
    Eigen::Matrix3d Rb = svd.matrixU() * w.transpose() * svd.matrixV().transpose();
    if (Rb.determinant() < 0) Rb = -Rb;

    std::vector<Eigen::Matrix3d> possible_R;
    possible_R.push_back(Ra);
    possible_R.push_back(Rb);
    std::vector<Eigen::Vector3d> possible_t;
    possible_t.push_back(ta);
    possible_t.push_back(tb);

    Eigen::Matrix4d result_transformation;
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            Eigen::Vector3d pt3d_in_frame1 = triangulate(pts0[1], pts1[1], possible_R[i], possible_t[j], K0, K1);
            Eigen::Vector3d pt3d_in_frame2 = possible_R[i] * pt3d_in_frame1 + possible_t[j];
            if (pt3d_in_frame1[2] > 0 && pt3d_in_frame2[2] > 0)
            {
                std::cout << "result relative:\n";
                std::cout << i << j  << std::endl;
                std::cout << possible_R[i] << std::endl << possible_t[j] << std::endl;
                result_transformation.block<3,3>(0,0) = possible_R[i];
                result_transformation.block<3,1>(0,3) = possible_t[j];
            }
        }
    }

    return result_transformation;
}

void read2dPoints(std::vector<point2d>& vec, std::string name)
{
    std::ifstream src;
    src.open(name);
    for (int i = 0; i < point_num; i++)
    {
        double x,y;
        src >> x >> y;
        vec.emplace_back(x, y, 1);
    }
    src.close();
}

void transformationError(Eigen::Matrix4d tr1, Eigen::Matrix4d tr2)
{
    Eigen::Matrix4d error_matrix = tr1 * tr2.inverse();

    Eigen::Vector3d eulerAngle = error_matrix.block<3,3>(0,0).eulerAngles(2,1,0);
    std::cout << "rotation_error:" << eulerAngle.transpose() << std::endl;//",sum:" << eulerAngle.no
    std::cout << "translation_error:" << error_matrix.block<3,1>(0,3).transpose() << ",sum:" << error_matrix.block<3,1>(0,3).norm() <<std::endl;

}

int main(int argc, char** argv)
{
    std::string pts3d_name = folder_name + "3dpts.txt";
    std::ifstream src;
    src.open(pts3d_name);

    point3d center3d = point3d::Zero();
    std::vector<point3d> pts3d;
    for (int i = 0; i < point_num; i++)
    {
        double x,y,z;
        src >> x >> y >> z;
        pts3d.emplace_back(x,y,z,1);
        center3d += point3d(x,y,z,1);
    }
    center3d = center3d / point_num;
    src.close();

    std::vector<Eigen::Matrix3d> intrinsics;
    std::vector<Eigen::Matrix4d> absolute_poses;
    std::vector<Eigen::Matrix4d> relative_poses;
    //-------------------------------- q2 -----------------------------------------------------

    for (int img_num = 1; img_num <= 3; ++img_num)
    {
        std::cout << "***********image" << img_num << "***********\n";
        std::string pt2d_name = folder_name + "img" + std::to_string(img_num) + ".txt";
        src.open(pt2d_name);

        // Read correspondence
        std::vector<match> matches;
        for (int i = 0; i < point_num; i++)
        {
            double x,y;
            src >> x >> y;
            matches.emplace_back(point2d(x,y,1), pts3d[i]);
        }
        src.close();

        // ----------Normalize the points ---------- //
        // 2d normalized matrix
        Eigen::Matrix3d normalized_matrix_2d;
        normalized_matrix_2d << 4032+3024, 0, 4032/2,
                                0, 4032+3024, 3024/2,
                                0, 0, 1;
        normalized_matrix_2d =  normalized_matrix_2d.inverse().eval();

        // 3d normalized matrix
        Eigen::Matrix3d covariance_matrix = Eigen::Matrix3d::Zero();
        for (match & cur_match: matches)
            covariance_matrix += ((cur_match.second - center3d) * (cur_match.second - center3d).transpose()).block<3,3>(0,0);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigenValue_solver(covariance_matrix);

        Eigen::Matrix3d eigenValue_matrix = Eigen::Matrix3d::Zero();
        eigenValue_matrix(0,0) = 1 / eigenValue_solver.eigenvalues()(0);
        eigenValue_matrix(1,1) = 1 / eigenValue_solver.eigenvalues()(1);
        eigenValue_matrix(2,2) = 1 / eigenValue_solver.eigenvalues()(2);

        Eigen::Matrix4d normalized_matrix_3d = Eigen::Matrix4d::Zero();
        normalized_matrix_3d.block<3,3>(0, 0) = eigenValue_solver.eigenvectors() * eigenValue_matrix * eigenValue_solver.eigenvectors().inverse();
        normalized_matrix_3d.block<3,1>(0, 3) = -normalized_matrix_3d.block<3,3>(0, 0) * center3d.block<3,1>(0,0);
        normalized_matrix_3d(3,3) = 1;
        // normalize points
        for (match &cur_match: matches)
        {
            cur_match.first = normalized_matrix_2d * cur_match.first;
            cur_match.second = normalized_matrix_3d * cur_match.second;
        }
        // ----------Normalize the points ---------- //

        // DLT
        Eigen::Matrix<double, Eigen::Dynamic, 12> constraint_matrix;
        constraint_matrix.resize(matches.size() * 2, 12);
        constraint_matrix.setZero();

        for (int i = 0; i < matches.size(); i++)
        {
            match &cur_match = matches[i];
            constraint_matrix.block<1,4>(i*2, 0) = cur_match.second.transpose();
            constraint_matrix.block<1,4>(i*2, 8) = -cur_match.first(0) * cur_match.second.transpose();
            constraint_matrix.block<1,4>(i*2+1, 4) = cur_match.second.transpose();
            constraint_matrix.block<1,4>(i*2+1, 8) = -cur_match.first(1) * cur_match.second.transpose();
        }

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(constraint_matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::Matrix<double, 3, 4> projection_matrix;
        projection_matrix << svd.matrixV()(0, 11), svd.matrixV()(1, 11), svd.matrixV()(2, 11), svd.matrixV()(3, 11),
                             svd.matrixV()(4, 11), svd.matrixV()(5, 11), svd.matrixV()(6, 11), svd.matrixV()(7, 11),
                             svd.matrixV()(8, 11), svd.matrixV()(9, 11), svd.matrixV()(10, 11), svd.matrixV()(11, 11);

        projection_matrix = normalized_matrix_2d.inverse() * projection_matrix * normalized_matrix_3d;

        std::cout << "projection_matrix:\n" <<  projection_matrix << std::endl;
        std::cout << "----------------" << std::endl;

        Eigen::Matrix3d camera_matrix = projection_matrix.block<3,3>(0, 0);

        // Decompose projection matrix to get intrinsic and pose
        Eigen::Matrix3d Rx = calculateR(camera_matrix, 0);
        camera_matrix = camera_matrix * Rx;
        Eigen::Matrix3d Ry = calculateR(camera_matrix, 1);
        camera_matrix = camera_matrix * Ry;
        Eigen::Matrix3d Rz = calculateR(camera_matrix, 2);
        camera_matrix = camera_matrix * Rz;

        Eigen::Matrix3d intrinsic = camera_matrix / camera_matrix(2,2);
        projection_matrix = projection_matrix / camera_matrix(2,2);

        std::cout << "Intrinsic:\n" << intrinsic << std::endl;
        std::cout << "----------------" << std::endl;

        Eigen::Matrix4d absolute_pose = Eigen::Matrix4d::Identity();
        absolute_pose.block<3,3>(0,0) = Rz.transpose() * Ry.transpose() * Rx.transpose();
        absolute_pose.block<3,1>(0,3) = intrinsic.inverse() * projection_matrix.block<3,1>(0,3);

        std::cout << "absolute_pose:\n";
        std::cout << absolute_pose.inverse() << std::endl;

        intrinsics.push_back(intrinsic);
        absolute_poses.push_back(absolute_pose.inverse());
    }

    std::cout <<"*****************q2 end******************\n";
    //-------------------------------- q3 -----------------------------------------------------

    for (int i = 0; i < absolute_poses.size(); i++)
    {
        std::cout << i << (i+1)%3 << "relative:\n";
        std::cout << absolute_poses[(i+1)%3].inverse() * absolute_poses[i] << std::endl;
        relative_poses.push_back(absolute_poses[(i+1)%3].inverse() * absolute_poses[i]);
    }


    std::vector<point2d> img1, img2, img3;

    std::string img1_name = folder_name + "img1.txt";
    std::string img2_name = folder_name + "img2.txt";
    std::string img3_name = folder_name + "img3.txt";

    read2dPoints(img1, img1_name);
    read2dPoints(img2, img2_name);
    read2dPoints(img3, img3_name);

    std::cout << "------computing essential 12\n";
    Eigen::Matrix3d essential_12 = computeEssentialMatrix(img1, img2, intrinsics[0], intrinsics[1]);
    Eigen::Matrix4d relative_12 = decomposeEssentialMatrix(essential_12, img1, img2, intrinsics[0], intrinsics[1]);

    std::cout << "---------computing essential 23\n";
    Eigen::Matrix3d essential_23 = computeEssentialMatrix(img2, img3, intrinsics[1], intrinsics[2]);
    Eigen::Matrix4d relative_23 = decomposeEssentialMatrix(essential_23, img2, img3, intrinsics[1], intrinsics[2]);

    std::cout << "---------computing essential 31\n";
    Eigen::Matrix3d essential_31 = computeEssentialMatrix(img3, img1, intrinsics[2], intrinsics[0]);
    Eigen::Matrix4d relative_31 = decomposeEssentialMatrix(essential_31, img3, img1, intrinsics[2], intrinsics[0]);

    transformationError(relative_poses[0], relative_12);
    transformationError(relative_poses[1], relative_23);
    transformationError(relative_poses[2], relative_31);
//    cv::Mat img1_cv(10, 2, CV_64F);
//    cv::Mat img2_cv(10, 2, CV_64F);
//    cv::Mat intrinsic0_cv, intrinsic1_cv;
//    cv::eigen2cv(intrinsics[0], intrinsic0_cv);
//    cv::eigen2cv(intrinsics[1], intrinsic1_cv);

//    for (int i = 0; i < 10; i++)
//    {
//        img1_cv.at<double>(i, 0) =  img1[i].x();
//        img1_cv.at<double>(i, 1) =  img1[i].y();
//        img2_cv.at<double>(i, 0) =  img2[i].x();
//        img2_cv.at<double>(i, 1) =  img2[i].y();
//    }

//    cv::Mat fundamental_matrix = cv::findFundamentalMat(img1_cv, img2_cv, cv::FM_8POINT);
//
//    cv::Mat intrinsic1_cv_transpose;
//    cv::transpose(intrinsic1_cv, intrinsic1_cv_transpose);
//    cv::Mat essential_matrix = intrinsic1_cv_transpose * fundamental_matrix * intrinsic1_cv;
//    std::cout << fundamental_matrix << std::endl;
//
//    std::cout << essential_matrix << std::endl;
//    std::cout << "-------------" << std::endl;

//    cv::Mat essential;
//    essential = cv::findEssentialMat(img1_cv, img2_cv, intrinsic1_cv);
//    std::cout << essential << std::endl;
//    cv::Mat r,t;
//    cv::recoverPose(essential, img1_cv, img2_cv, intrinsic1_cv, r,t);
//    std::cout << r << "\n" << t << std::endl;



}