#include<cstring>
#include<iostream>
#include<fstream>
#include<cmath>
#include<vector>
#include<ctime>

#include<Eigen/Core>
#include<Eigen/Geometry>

#include<open3d/Open3D.h>

typedef Eigen::Vector2d point2d;

template <typename Type>
using AlignedVector = std::vector<Type>;
typedef AlignedVector<point2d> Pointcloud;


std::string data_path = "../../material/hw1_data/data.txt";
int total_frame_num = 360;
int total_ring_num = 360;
double resolution = 2.0 * M_PI / total_ring_num;
int max_iterations = 100;

std::clock_t start, end;


double euclideanDistance(point2d point1, point2d point2)
{
    return sqrt(pow(point1.x() - point2.x(), 2) + pow(point1.y() - point2.y(), 2));
}

void visualPointCloud(Pointcloud& cloud0,Pointcloud& cloud1)
{
    auto cloud0_ptr = std::make_shared<open3d::geometry::PointCloud>();
    auto cloud1_ptr = std::make_shared<open3d::geometry::PointCloud>();
    for (size_t i = 0; i < cloud0.size(); i++)
    {
        point2d& point = cloud0[i];
        cloud0_ptr->colors_.emplace_back(1,0,0);
        cloud0_ptr->points_.emplace_back(point.x(), point.y(), 0);
    }

    for (size_t i = 0; i < cloud1.size(); i++)
    {
        point2d& point = cloud1[i];
        cloud1_ptr->colors_.emplace_back(0,0,1);
        cloud1_ptr->points_.emplace_back(point.x(), point.y(), 0);
    }
    open3d::visualization::DrawGeometries({cloud0_ptr, cloud1_ptr});

}

int nearestSearch_brute(point2d query, Pointcloud list)
{
    double distance = MAXFLOAT;
    int idx = -1;

    for (size_t i = 0; i < list.size(); i++)
    {
        point2d& neighbour = list[i];

        double cur_distance = euclidianDistance(query, neighbour);
        if (cur_distance < distance)
        {
            distance = cur_distance;
            idx = i;
        }
    }
    return idx;
}

void transformPointCloud(Pointcloud& cloud, Eigen::Matrix2d R, Eigen::Vector2d t)
{
    for (size_t i = 0; i < cloud.size(); i++)
    {
        cloud[i] = R * cloud[i] + t;
    }
}

void myicp(Pointcloud ref_cloud, Pointcloud aligned_cloud, Eigen::Matrix2d& R, Eigen::Vector2d& t)
{
    int iter = 0;

    transformPointCloud(aligned_cloud, R, t);
    while (iter < max_iterations)
    {
        // visualPointCloud(ref_cloud, aligned_cloud);
        // Create correspondence set and get the centroid of the two sets, compute loss
        std::vector<std::pair<int,int>> idx_pair_list;
        point2d alinged_centroid ,ref_centroid ;

        for (size_t aligned_idx = 0; aligned_idx < aligned_cloud.size(); aligned_idx++)
        {
            int ref_idx = nearestSearch_brute(aligned_cloud[aligned_idx], ref_cloud);
            idx_pair_list.emplace_back(aligned_idx,ref_idx);

            alinged_centroid += aligned_cloud[aligned_idx]; 
            ref_centroid += ref_cloud[ref_idx]; 
        }

        alinged_centroid /= aligned_cloud.size();
        ref_centroid /= aligned_cloud.size();

        // create association matrix 
        Eigen::Matrix2d covaraince_matrix = Eigen::Matrix2d::Zero();
        for (size_t i = 0; i < idx_pair_list.size(); i++)
        {
            point2d aligned_point = aligned_cloud[idx_pair_list[i].first];
            point2d ref_point = ref_cloud[idx_pair_list[i].second];
             
            aligned_point -= alinged_centroid;
            ref_point -= ref_centroid;

            covaraince_matrix += ref_point*aligned_point.transpose();
        }

        // solve R and t
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(covaraince_matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::Matrix2d delta_R = svd.matrixU() * (svd.matrixV().transpose());
        if (delta_R.determinant() < 0)
            delta_R = -delta_R;
        Eigen::Vector2d delta_t = ref_centroid - delta_R * alinged_centroid;

        // determine convergence
        double R_change = abs(delta_R(1,0));
        double t_change = delta_t.norm();

        // std::cout << "R_change:" << R_change << ", t_change:" <<  t_change << std::endl;

        if (R_change < 10e-5 && t_change < 10e-3)
        {
            std::cout << "icp converge! R_change is " << R_change << ", t_change is " << t_change << ", using " << iter << " iter" << std::endl;
            std::cout << "R: " << R << std::endl;
            std::cout << "t: " << t.transpose() << std::endl;
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

int main(int argc, char** argv)
{

    start = std::clock();
    std::ifstream src;
    src.open(data_path);

    std::vector<Pointcloud> all_frames;
    all_frames.reserve(total_frame_num);
    for (int frame_i = 0; frame_i < total_frame_num; frame_i++)
    {
        Pointcloud& cur_cloud = all_frames[frame_i];
        for (int ring_i = 0; ring_i < total_ring_num; ring_i++)  
        {
            float distance;
            src >> distance;

            cur_cloud.emplace_back(std::cos((ring_i + 1) * resolution) * distance,
                                   std::sin((ring_i + 1) * resolution) * distance);
        }
    }
    src.close();
    
    Eigen::Matrix2d global_pose_R = Eigen::Matrix2d::Identity();
    Eigen::Vector2d global_pose_t = Eigen::Vector2d::Zero();

    Pointcloud global_trajactory;
    Pointcloud empty;
    global_trajactory.emplace_back(0,0);   

    for (int curFrame_id = 1; curFrame_id < total_frame_num; curFrame_id++)
    {
        Pointcloud prev_cloud = all_frames[curFrame_id-1];
        Pointcloud cur_cloud = all_frames[curFrame_id];
        Eigen::Matrix2d R = Eigen::Matrix2d::Identity();
        Eigen::Vector2d t = Eigen::Vector2d::Zero();

        myicp(prev_cloud, cur_cloud, R, t);
        // visualPointCloud(prev_cloud, cur_cloud);

        global_pose_R = R * global_pose_R;
        global_pose_t = R * global_pose_t + t;

        global_trajactory.emplace_back(global_pose_t.x(), global_pose_t.y());   

        // std::cout << global_pose_t.x() << " " << global_pose_t.y() << std::endl; 
        // if (curFrame_id == 2) break;
    }

    
    visualPointCloud(global_trajactory, global_trajactory);

}