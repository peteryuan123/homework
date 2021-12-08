//
// Created by mpl on 2021/10/5.
//
#include <cstring>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <ctime>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <ros/ros.h>
#include <ros/publisher.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>

typedef Eigen::Vector2f point;
typedef std::vector<point, Eigen::aligned_allocator<point>> PointCloud;

int total_frame_num = 360;
int total_ring_num = 360;
float resolution = 2.0 * M_PI / total_ring_num;
int max_iterations = 100;

std::clock_t start,end;

void readFrame(PointCloud& frame, std::ifstream& src)
{
    for (int i = 0; i < total_ring_num; i++)
    {
        float distance;
        src >> distance;

        frame.emplace_back(std::cos((i + 1) * resolution) * distance,
                           std::sin((i + 1) * resolution) * distance);
    }

}

inline float euclideanDistance(point point1, point point2)
{
    return sqrt(pow(point1.x() - point2.x(), 2) + pow(point1.y() - point2.y(), 2));
}

int nearestSearch_brute(point& query, PointCloud& list)
{
    float distance = MAXFLOAT;
    int idx = 0;

    for (size_t i = 0; i < list.size(); i++)
    {
        point& neighbour = list[i];

        float cur_distance = euclideanDistance(query, neighbour);
        if (cur_distance < distance)
        {
            distance = cur_distance;
            idx = i;
        }
    }

    if (distance > 0.5) return -1;
    return idx;
}

void transformPointCloud(PointCloud & cloud, Eigen::Matrix2f& R, Eigen::Vector2f& t)
{
    for (size_t i = 0; i < cloud.size(); i++)
    {
        cloud[i] = R * cloud[i] + t;
    }
}

void myIcp(PointCloud ref_cloud, PointCloud aligned_cloud, Eigen::Matrix2f& R, Eigen::Vector2f& t)
{
    int iter = 0;

    transformPointCloud(aligned_cloud, R, t);
    while (iter < max_iterations)
    {
        std::vector<std::pair<size_t,size_t>> idx_pair_list;
        point aligned_centroid = point::Zero();
        point ref_centroid = point::Zero();
        for (size_t aligned_idx = 0; aligned_idx < aligned_cloud.size(); aligned_idx++)
        {
            int ref_idx = nearestSearch_brute(aligned_cloud[aligned_idx], ref_cloud);
            if (ref_idx < 0) continue;
            idx_pair_list.emplace_back(aligned_idx,ref_idx);

            aligned_centroid += aligned_cloud[aligned_idx];
            ref_centroid += ref_cloud[ref_idx];
        }

        aligned_centroid /= aligned_cloud.size();
        ref_centroid /= aligned_cloud.size();

        // create covariance matrix
        Eigen::Matrix2f covariance_matrix = Eigen::Matrix2f::Zero();
        for (size_t i = 0; i < idx_pair_list.size(); i++)
        {
            point aligned_point = aligned_cloud[idx_pair_list[i].first];
            point ref_point = ref_cloud[idx_pair_list[i].second];

            aligned_point -= aligned_centroid;
            ref_point -= ref_centroid;

            covariance_matrix += ref_point*aligned_point.transpose();
        }

        // solve R and t
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(covariance_matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::Matrix2f delta_R = svd.matrixU() * (svd.matrixV().transpose());
        if (delta_R.determinant() < 0)
            delta_R = -delta_R;
        Eigen::Vector2f delta_t = ref_centroid - delta_R * aligned_centroid;

        // determine convergence
        double R_change = abs(delta_R(1,0));
        double t_change = delta_t.norm();

        // std::cout << "R_change:" << R_change << ", t_change:" <<  t_change << std::endl;

        if (R_change < 10e-5 && t_change < 10e-3)
        {
//            std::cout << "icp converge! R_change is " << R_change << ", t_change is " << t_change << ", using " << iter << " iter" << std::endl;
//            std::cout << "R: " << R << std::endl;
//            std::cout << "t: " << t.transpose() << std::endl;
            return;
        }

        // update pose
        R = delta_R * R;
        t = delta_R * t + delta_t;

        transformPointCloud(aligned_cloud, delta_R, delta_t);

        iter++;
    }

    std::cout << "max iteration reached! " << std::endl;
    std::cout << "R: \n" << R << std::endl;
    std::cout << "t: " << t.transpose() << std::endl;

}

void eigenPoseToGeometryPose(geometry_msgs::PoseStamped & pose, Eigen::Matrix2f& R, Eigen::Vector2f& t)
{
    pose.pose.position.x = t.x();
    pose.pose.position.y = t.y();
    pose.pose.position.z = 1;

    float theta = std::atan2(R(1,0), R(0,0));

    Eigen::Quaternionf orientation = Eigen::AngleAxisf(theta, Eigen::Vector3f::UnitZ()) *
                                     Eigen::AngleAxisf(0, Eigen::Vector3f::UnitY()) *
                                     Eigen::AngleAxisf(0, Eigen::Vector3f::UnitX());

    pose.pose.orientation.x = orientation.x();
    pose.pose.orientation.y = orientation.y();
    pose.pose.orientation.z = orientation.z();
    pose.pose.orientation.w = orientation.w();

    pose.header.frame_id = "map";
    pose.header.stamp = ros::Time::now();
}



int main(int argc, char** argv)
{
    ros::init(argc, argv, "icp");
    ros::NodeHandle nh;

    std::string data_path = argv[1];
    start = std::clock();
    
    ros::Publisher posePub;
    ros::Publisher pathPub;
    posePub = nh.advertise<nav_msgs::Odometry>("pose", 400);
    pathPub = nh.advertise<nav_msgs::Path>("path", 400);

    PointCloud lastFrame;
    PointCloud curFrame;

    std::ifstream src;
    src.open(data_path);

    geometry_msgs::PoseStamped global_pose;
    nav_msgs::Path path;
    path.header.frame_id = "map";

    Eigen::Matrix2f global_pose_R = Eigen::Matrix2f::Identity();
    Eigen::Vector2f global_pose_t = Eigen::Vector2f::Zero();

    eigenPoseToGeometryPose(global_pose, global_pose_R, global_pose_t);
    path.poses.push_back(global_pose);
    path.header.stamp = ros::Time::now();
    pathPub.publish(path);

    readFrame(lastFrame, src);

    for (int i = 0; i < total_frame_num - 1; i++)
    {
        readFrame(curFrame, src);

        Eigen::Matrix2f R = Eigen::Matrix2f::Identity();
        Eigen::Vector2f t = Eigen::Vector2f::Zero();
        myIcp(lastFrame, curFrame, R, t);

        global_pose_R = R * global_pose_R;
        global_pose_t = R * global_pose_t + t;

        eigenPoseToGeometryPose(global_pose, global_pose_R, global_pose_t);

        path.poses.push_back(global_pose);
        path.header.stamp = ros::Time::now();
        pathPub.publish(path);

        lastFrame = std::move(curFrame);
    }

    end = std::clock();

    std::cout << "orientation:" << std::atan2(global_pose_R(1,0), global_pose_R(0,0)) / (2*M_PI) * 360 << std::endl;
    std::cout << "translation:" << global_pose_t.norm() << std::endl;
    std::cout << "time: " << (double)(end-start)/CLOCKS_PER_SEC << std::endl;
}