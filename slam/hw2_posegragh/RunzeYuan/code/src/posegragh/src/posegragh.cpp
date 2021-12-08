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
#include <nav_msgs/Path.h>

#include <g2o/types/slam2d/types_slam2d.h>
#include <g2o/core/block_solver.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>


typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolverType;
typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolveType;


typedef Eigen::Matrix3d poseMatrix;
typedef Eigen::Vector3d poseVector;
typedef std::vector<poseMatrix, Eigen::aligned_allocator<poseMatrix>> Trajectory;

int total_num_poses = 12;

std::clock_t start,end;

void readPose(Trajectory& relative_poses, std::ifstream& src)
{
    std::string delta_x_string;
    std::string delta_y_string;
    std::string delta_theta_string;

    std::getline(src,delta_x_string);
    std::getline(src,delta_y_string);
    std::getline(src,delta_theta_string);

    std::stringstream x_stream(delta_x_string);
    std::stringstream y_stream(delta_y_string);
    std::stringstream theta_stream(delta_theta_string);

    for (int i = 0; i < total_num_poses; i++)
    {
        float delta_x, delta_y, delta_theta;
        x_stream >> delta_x;    y_stream >> delta_y;    theta_stream >>delta_theta;

        poseMatrix relative_pose;
        relative_pose << std::cos(delta_theta), -std::sin(delta_theta), delta_x,
                         std::sin(delta_theta), std::cos(delta_theta), delta_y,
                         0, 0, 1;

        relative_poses.push_back(relative_pose);
    }

}

inline poseMatrix v2t(poseVector poseV)
{
    poseMatrix poseM;
    poseM << std::cos(poseV.z()), -std::sin(poseV.z()), poseV.x(),
             std::sin(poseV.z()), std::cos(poseV.z()), poseV.y(),
             0, 0, 1;
    return poseM;
}

inline poseVector t2v(poseMatrix& poseM)
{
    poseVector poseV;
    poseV.x() = poseM(0, 2);
    poseV.y() = poseM(1, 2);;
    poseV.z() = std::atan2(poseM(1,0), poseM(0,0));
    return poseV;
}

void eigenPoseToGeometryPose(geometry_msgs::PoseStamped & pose_msg, poseMatrix &pose)
{
    pose_msg.pose.position.x = pose(0,2);
    pose_msg.pose.position.y = pose(1,2);
    pose_msg.pose.position.z = 1;

    float theta = std::atan2(pose(1,0), pose(0,0));

    Eigen::Quaternionf orientation = Eigen::AngleAxisf(theta, Eigen::Vector3f::UnitZ()) *
                                     Eigen::AngleAxisf(0, Eigen::Vector3f::UnitY()) *
                                     Eigen::AngleAxisf(0, Eigen::Vector3f::UnitX());

    pose_msg.pose.orientation.x = orientation.x();
    pose_msg.pose.orientation.y = orientation.y();
    pose_msg.pose.orientation.z = orientation.z();
    pose_msg.pose.orientation.w = orientation.w();

    pose_msg.header.frame_id = "map";
    pose_msg.header.stamp = ros::Time::now();
}

void publishPoses(Trajectory &trajectory, ros::Publisher& pub)
{
    ros::Rate loop_rate(10);

    nav_msgs::Path path;
    for (poseMatrix& pose: trajectory)
    {
        geometry_msgs::PoseStamped pose_msg;
        eigenPoseToGeometryPose(pose_msg, pose);
        path.header.frame_id = "map";
        path.poses.push_back(pose_msg);
        path.header.stamp = ros::Time::now();
        loop_rate.sleep();
        pub.publish(path);
    }
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "pose_gragh");
    ros::NodeHandle nh;

    ros::Publisher originPathPub;
    ros::Publisher optimizedPathPub;
    ros::Publisher finalPathPub;
    originPathPub = nh.advertise<nav_msgs::Path>("nonOptimizedPose", 400);
    optimizedPathPub = nh.advertise<nav_msgs::Path>("optimizedPose", 400);
    finalPathPub = nh.advertise<nav_msgs::Path>("finalPose", 400);

    start = std::clock();

    // Open file
    std::string data_path = argv[1];
    std::ifstream src;
    std::ofstream dest;
    src.open(data_path);

    // Read relative poses
    Trajectory relative_poses;
    readPose(relative_poses, src);

    // Get global poses using relative poses
    Trajectory global_poses;
    poseMatrix cur_pose = poseMatrix::Identity();
    cur_pose(0,2) = 1;
    global_poses.push_back(cur_pose);
    for (int i = 0; i < total_num_poses; i++)
    {
        cur_pose = cur_pose * relative_poses[i];
        std::cout << cur_pose << std::endl;
        global_poses.push_back(cur_pose);
    }

    // Publish origin poses
    publishPoses(global_poses, originPathPub);
    dest.open("/home/mpl/homework/slam/hw2_posegragh/posegraph_ws/trajectory_dead_reckoning.txt");
    for (int i = 0; i < total_num_poses; i++)
    {
        poseVector pose = t2v(global_poses[i]);
        dest << pose[0] << " " << pose[1] << " " << pose[2] << std::endl;
    }
    src.close();
    dest.close();
    //----------------------------------- Q2-----------------------------------------------//

    // set up the optimizer
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolveType>())
            );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);
    // set the fixed vertex
    g2o::VertexSE2 *vertex = new g2o::VertexSE2();
    vertex->setId(0);
    vertex->setEstimate(g2o::SE2(1,0,0));
    vertex->setFixed(true);
    optimizer.addVertex(vertex);

    // adding vertex
    for (int i = 1; i < total_num_poses; i++)
    {
        g2o::VertexSE2 *vertex = new g2o::VertexSE2();
        vertex->setId(i);
        vertex->setEstimate(g2o::SE2(t2v(global_poses[i])));
        optimizer.addVertex(vertex);
    }

    // add edge
    for (int i = 0; i < total_num_poses; i++)
    {
        g2o::EdgeSE2* edge = new g2o::EdgeSE2();
        int src_id = i, dest_id = (i+1)%total_num_poses;
        edge->setId(i);
        edge->setVertex(0, optimizer.vertex(src_id));
        edge->setVertex(1, optimizer.vertex(dest_id));
        edge->setMeasurement(g2o::SE2(t2v(relative_poses[i])));
        edge->setInformation(Eigen::Matrix3d::Identity());
        optimizer.addEdge(edge);
    }

    // optimize
    std::cout<<"optimizing pose graph, vertices: "<< optimizer.vertices().size() << std::endl;
    optimizer.save("/home/mpl/homework/slam/hw2_posegragh/posegraph_ws/result_before.g2o");
    optimizer.initializeOptimization();
    optimizer.optimize(20);
    optimizer.save( "/home/mpl/homework/slam/hw2_posegragh/posegraph_ws/result_loop.g2o" );
    // visualize
    Trajectory optimized_poses;
    for (int i = 0; i < optimizer.vertices().size(); i++)
    {
        double result_pose[3];
        optimizer.vertex(i)->getEstimateData(result_pose);
        optimized_poses.push_back(v2t(poseVector(result_pose)));
    }
    optimized_poses.push_back(v2t(poseVector(1,0,0)));
    publishPoses(optimized_poses, optimizedPathPub);

    dest.open("/home/mpl/homework/slam/hw2_posegragh/posegraph_ws/trajectory_loop_closure.txt");
    for (int i = 0; i < total_num_poses; i++)
    {
        poseVector pose = t2v(optimized_poses[i]);
        dest << pose[0] << " " << pose[1] << " " << pose[2] << std::endl;
    }
    dest.close();
    //----------------------------------- Q3-----------------------------------------------//

    // read more constraints
    std::string constraints_path = argv[2];
    src.open(constraints_path);
    Trajectory relative_pose_two_frames;
    readPose(relative_pose_two_frames, src);
    src.close();

    // add more constraints
    for (int i = 0; i < total_num_poses - 2; i++)
    {
        g2o::EdgeSE2* edge = new g2o::EdgeSE2();

        edge->setId(i + 1 + optimizer.edges().size());
        edge->setVertex(0, optimizer.vertex(i));
        edge->setVertex(1, optimizer.vertex(i + 2));
        edge->setMeasurement(g2o::SE2(t2v(relative_pose_two_frames[i])));
        edge->setInformation(Eigen::Matrix3d::Identity());
        optimizer.addEdge(edge);
    }

    optimizer.save( "/home/mpl/homework/slam/hw2_posegragh/posegraph_ws/result_loop_before_add_constraints.g2o" );
    std::cout<<"optimizing pose graph, vertices: "<< optimizer.vertices().size() << std::endl;
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    optimizer.save( "/home/mpl/homework/slam/hw2_posegragh/posegraph_ws/result_loop_after_add_constraints.g2o" );
    std::cout<<"Optimization done."<< std::endl;

    // visualize
    Trajectory final_poses;
    for (int i = 0; i < optimizer.vertices().size(); i++)
    {
        double result_pose[3];
        optimizer.vertex(i)->getEstimateData(result_pose);
        final_poses.push_back(v2t(poseVector(result_pose)));
        std::cout << poseVector(result_pose).transpose() << std::endl;
    }
    final_poses.push_back(v2t(poseVector(1,0,0)));
    publishPoses(final_poses, finalPathPub);

    dest.open("/home/mpl/homework/slam/hw2_posegragh/posegraph_ws/trajectory_more_constraints.txt");
    for (int i = 0; i < total_num_poses; i++)
    {
        poseVector pose = t2v(final_poses[i]);
        dest << pose[0] << " " << pose[1] << " " << pose[2] << std::endl;
    }
    dest.close();

    optimizer.clear();


    end = std::clock();

}