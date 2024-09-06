#include "tools.hpp"
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <random>
#include <ctime>
#include <iostream>
#include <fstream>
#include <sstream>
#include "bavoxel.hpp"
#include <malloc.h>
#include <windows.h>
#include <yaml-cpp/yaml.h>
#include <map>
#include <variant>
#include <vector>
#include <stdio.h>
#include <direct.h>
#include <sys/types.h>
#include <filesystem>
#include <utility>
#include <thread>
#include <chrono>
#include "ThreadPool.hpp"

#include <omp.h> // OpenMP, 한 PC에서 병렬 프로그래밍

//#include <pcl/visualization/could_viewer.h>

#include <thrust/system/omp/execution_policy.h>
#include <thrust/system/omp/vector.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/sort.h>

//#include <cupoch/cupoch.h>
//#include "cupoch/utility/console.h"
//#include "cupoch/utility/device_vector.h"
//#include <cupoch/registration/colored_icp.h>
//#include <cupoch/registration/filterreg.h>
//#include <cupoch/geometry/kdtree_flann.h>

using namespace std;
using namespace YAML;
//using namespace cupoch;
//using namespace geometry;
namespace fs = filesystem;
using Hyperparameters = map<std::string, std::variant<int, float, std::string>>;

void save_point_cloud(pcl::PointCloud<PointType>& pl, const std::string& filename)
{
    pcl::PLYWriter writer;
    writer.write(filename, pl, false);
    std::cout << "Point cloud saved to: " << filename << std::endl; // 저장 경로 출력
}

template <typename T>
void pub_pl_func(T& pl, const std::string& filename = "")
{
    pl.height = 1;
    pl.width = pl.size();

    if (!filename.empty()) {
        save_point_cloud(pl, filename);
    }
}


int read_pose(vector<double>& tims, PLM(3)& rots, PLV(3)& poss, string prename, int num)
{
    string readname = prename + "alidarPose.csv";

    cout << readname << endl;
    ifstream inFile(readname);

    if (!inFile.is_open())
    {
        printf("open fail\n"); return 0;
    }

    int pose_size = 0;
    string lineStr, str;
    Eigen::Matrix4d aff;
    vector<double> nums;

    int ord = 0;
    int cnt = 0;
    while (getline(inFile, lineStr))
    {
        if (num * 4 == cnt) {
            return pose_size;
        };

        ord++;
        stringstream ss(lineStr);
        while (getline(ss, str, ','))
            nums.push_back(stod(str));

        if (ord == 4)
        {
            for (int j = 0; j < 16; j++)
                aff(j) = nums[j];

            Eigen::Matrix4d affT = aff.transpose();

            rots.push_back(affT.block<3, 3>(0, 0));
            poss.push_back(affT.block<3, 1>(0, 3));
            tims.push_back(affT(3, 3));
            nums.clear();
            ord = 0;
            pose_size++;
        }
        cnt++;
    }

    return pose_size;
}

void read_file(vector<IMUST>& x_buf, vector<pcl::PointCloud<PointType>::Ptr>& pl_fulls, string& prename, int num)
{
    PLV(3) poss; PLM(3) rots;
    vector<double> tims;
    int pose_size = read_pose(tims, rots, poss, prename, num);

    #pragma omp parallel for
    for (int m = 0; m < pose_size; m++)
    {
        string filename = prename + "full" + to_string(m) + ".pcd";

        pcl::PointCloud<PointType>::Ptr pl_ptr(new pcl::PointCloud<PointType>());
        pcl::PointCloud<pcl::PointXYZ> pl_tem;
        pcl::io::loadPCDFile(filename, pl_tem);
        for (pcl::PointXYZ& pp : pl_tem.points)
        {
            PointType ap; // PointXYZINormal PointType
            ap.x = pp.x; ap.y = pp.y; ap.z = pp.z;
            ap.intensity = 1; // 임의값
            // ap.intensity = pp.intensity;
            // ap.normal_x = pp.normal_x;
            // ap.normal_y = pp.normal_y;
            // ap.normal_z = pp.normal_z;
            pl_ptr->push_back(ap);
        }

        pl_fulls.push_back(pl_ptr);

        IMUST curr;
        curr.R = rots[m]; curr.p = poss[m]; curr.t = tims[m];
        x_buf.push_back(curr);
    }
}

void data_show(vector<IMUST> x_buf, vector<pcl::PointCloud<PointType>::Ptr>& pl_fulls, string savebase = "/home/jw/")
{
    IMUST es0 = x_buf[0];
    #pragma omp parallel for
    for (uint i = 0; i < x_buf.size(); i++)
    {
        x_buf[i].p = es0.R.transpose() * (x_buf[i].p - es0.p);
        x_buf[i].R = es0.R.transpose() * x_buf[i].R;
    }

    pcl::PointCloud<PointType> pl_send, pl_path;
    int winsize = x_buf.size();
    string outtext = "";
    for (int i = 0; i < winsize; i++)
    {
        pcl::PointCloud<PointType> pl_tem = *pl_fulls[i];
        down_sampling_voxel(pl_tem, 0.05);
        pl_transform(pl_tem, x_buf[i]);
        pl_send += pl_tem;

        if ((i % 100 == 0 && i != 0) || i == winsize - 1)
        {
            string save_name = savebase + to_string(i) + ".ply";
            pub_pl_func(pl_send, save_name);
            pl_send.clear();
            this_thread::sleep_for(std::chrono::milliseconds(500));  // sleep 대체
        }

        PointType ap;
        ap.x = x_buf[i].p.x();
        ap.y = x_buf[i].p.y();
        ap.z = x_buf[i].p.z();
        ap.curvature = i;

        // write rots and poss to txt file "pose.txt"
        // x_buf[i].R, x_buf[i].p
        outtext += to_string(x_buf[i].R(0, 0)) + "," + to_string(x_buf[i].R(0, 1)) + "," + to_string(x_buf[i].R(0, 2)) + "," + to_string(x_buf[i].p(0)) + "\n";
        outtext += to_string(x_buf[i].R(1, 0)) + "," + to_string(x_buf[i].R(1, 1)) + "," + to_string(x_buf[i].R(1, 2)) + "," + to_string(x_buf[i].p(1)) + "\n";
        outtext += to_string(x_buf[i].R(2, 0)) + "," + to_string(x_buf[i].R(2, 1)) + "," + to_string(x_buf[i].R(2, 2)) + "," + to_string(x_buf[i].p(2)) + "\n";
        outtext += "0,0,0," + to_string(i) + "\n";

        pl_path.push_back(ap);
    }
    std::ofstream file(savebase + "pose.csv");
    if (file.is_open())
    {
        file << outtext;
        file.close();
    }

    pub_pl_func(pl_path, savebase + "path.ply");
}

void optimize_memory() {
    HANDLE heap = GetProcessHeap();
    if (heap) {
        PROCESS_HEAP_ENTRY entry;
        entry.lpData = NULL;

        SIZE_T heapSizeBefore = 0;
        while (HeapWalk(heap, &entry)) {
            if ((entry.wFlags & PROCESS_HEAP_ENTRY_BUSY) != 0) {
                heapSizeBefore += entry.cbData;
            }
        }
        if (GetLastError() != ERROR_NO_MORE_ITEMS) {
            std::cerr << "HeapWalk failed with error: " << GetLastError() << std::endl;
        }

        std::cout << "Heap size before optimization: " << heapSizeBefore << " bytes" << std::endl;

        BOOL result = HeapCompact(heap, 0);
        if (!result) {
            std::cerr << "HeapCompact failed with error: " << GetLastError() << std::endl;
        }

        SIZE_T heapSizeAfter = 0;
        entry.lpData = NULL;
        while (HeapWalk(heap, &entry)) {
            if ((entry.wFlags & PROCESS_HEAP_ENTRY_BUSY) != 0) {
                heapSizeAfter += entry.cbData;
            }
        }
        if (GetLastError() != ERROR_NO_MORE_ITEMS) {
            std::cerr << "HeapWalk failed with error: " << GetLastError() << std::endl;
        }

        std::cout << "Heap size after optimization: " << heapSizeAfter << " bytes" << std::endl;
    }
    else {
        std::cerr << "Failed to get process heap with error: " << GetLastError() << std::endl;
    }
}

void createDirectory(const string& directory_path) {
    error_code ec;
    if (filesystem::create_directories(directory_path, ec)) {
        cout << "Directories created successfully." << directory_path << endl;
    }
    else {
        if (ec) {
            std::cout << "Error: " << ec.message() << std::endl;
        }
        else {
            //std::cout << "Directories already exist." << std::endl;
        }
    }
}

void loadHyperparameters(const std::string& filename, Hyperparameters& params) {
    try {
        YAML::Node config = YAML::LoadFile(filename);
        if (!config) {
            throw std::runtime_error("Failed to load YAML file: " + filename);
        }
        for (YAML::const_iterator it = config.begin(); it != config.end(); ++it) {
            std::string key = it->first.as<std::string>();
            std::string type = it->second["type"].as<std::string>();
            if (type == "float") {
                params[key] = it->second["value"].as<float>();
            }
            else if (type == "int") {
                params[key] = it->second["value"].as<int>();
            }
            else if (type == "string") {
                params[key] = it->second["value"].as<std::string>();
            }
            else {
                throw std::runtime_error("Invalid type for key: " + key);
            }
        }
    }
    catch (const YAML::BadFile& e) {
        std::cerr << "YAML::BadFile exception caught: Failed to open or read the YAML file: " << filename << std::endl;
        std::cerr << "Error message: " << e.what() << std::endl;
        throw; // 재던지기: 프로그램이 계속 진행되기를 원하지 않을 경우 사용
    }
    catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        throw; // 다른 예외도 다시 던질 수 있습니다.
    }
}

int main(int argc, char** argv)
{
    cout << "============================================================================================" << endl;
    cout << "================================= START 3D RECONSTRUCTION! =================================" << endl;
    cout << "============================================================================================" << endl;

    /* READ YAML FILE */
    Hyperparameters params;
    try {
        loadHyperparameters("..\\hyperparameters.yaml", params);
    }
    catch (const std::exception& e) {
        std::cerr << "Failed to load hyperparameters: " << e.what() << std::endl;
        return 1;
    }

    // shows params
    for (const auto& [key, value] : params) {
        std::cout << "Key: " << key << " -> Value: ";
        if (std::holds_alternative<std::string>(value)) {
            std::cout << std::get<std::string>(value);
        }
        else if (std::holds_alternative<int>(value)) {
            std::cout << std::get<int>(value);
        }
        else if (std::holds_alternative<float>(value)) {
            std::cout << std::get<float>(value);
        }
        else {
            std::cout << "Unknown type";
        }
        std::cout << std::endl;
    }

    /* SET PARAMETERS */
    std::string BASE_DIR = std::get<string>(params["BASE_DIR"]);
    std::string scene_name = std::get<string>(params["SCENE_NAME"]);
    std::string sourceDir = BASE_DIR + "\\" + scene_name + "\\";
    std::string SAVE_DIR = std::get<string>(params["SAVE_DIR"]);
    
    std::string DO_PARALLEL = std::get<string>(params["DO_PARALLEL"]);
    float voxel_size = std::get<float>(params["VOXEL_SIZE"]);
    std::string EIGEN_VALUE_ARRAY_STRING = std::get<string>(params["EIGEN_VALUE_ARRAY"]);
    int limit_frame = std::get<int>(params["LIMIT_FRAME"]);

    /* CHANGE HYPERPARAMETERS */
    set_voxel_size(voxel_size); // hyper parameter voxel_size 변경
    std::vector<float> eigenValues = parseEigenValueArray(EIGEN_VALUE_ARRAY_STRING);
    set_eigen_value_array(eigenValues);

    /* thread pool */
    ThreadPool pool(thread::hardware_concurrency());

    vector<IMUST> x_buf;
    vector<pcl::PointCloud<PointType>::Ptr> pl_fulls;

    createDirectory(SAVE_DIR);
    
    cout << "We limit frame num : " << limit_frame << endl;

    auto time01 = std::chrono::high_resolution_clock::now();
    read_file(x_buf, pl_fulls, sourceDir, limit_frame);
    auto time02 = std::chrono::high_resolution_clock::now();

    // calculate time
    IMUST es0 = x_buf[0];
    #pragma omp parallel for
    for (uint i = 0; i < x_buf.size(); i++)
    {
        x_buf[i].p = es0.R.transpose() * (x_buf[i].p - es0.p);
        x_buf[i].R = es0.R.transpose() * x_buf[i].R;
    }

    win_size = x_buf.size();
    int total_size = x_buf.size();
    // win_size = 20;
    printf("The size of poses: %d\n", total_size);
    

    auto time03 = std::chrono::high_resolution_clock::now();
    createDirectory(SAVE_DIR + "\\before");
    data_show(x_buf, pl_fulls, SAVE_DIR + "\\before");
    printf("Check the point cloud with the initial poses.\n");
    //printf("If no problem, input '1' to continue or '0' to exit...\n");
    auto time04 = std::chrono::high_resolution_clock::now();
    //int a; cin >> a; if (a == 0) exit(0);
    auto time041 = std::chrono::high_resolution_clock::now();

    pcl::PointCloud<PointType> pl_full, pl_surf, pl_path, pl_send;

    std::chrono::high_resolution_clock::time_point intime01;
    std::chrono::high_resolution_clock::time_point intime02;
    std::chrono::high_resolution_clock::time_point intime03;
    std::chrono::high_resolution_clock::time_point intime04;
    std::chrono::high_resolution_clock::time_point intime05;
    std::chrono::high_resolution_clock::time_point intime06;
    std::chrono::high_resolution_clock::time_point intime07;

    std::chrono::duration<double> avg_intime01;
    std::chrono::duration<double> avg_intime02;
    std::chrono::duration<double> avg_intime03;
    std::chrono::duration<double> avg_intime04;
    std::chrono::duration<double> avg_intime05;
    std::chrono::duration<double> avg_intime06;

    for (int iterCount = 0; iterCount < 1; iterCount++)
    {
        unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;

        intime01 = std::chrono::high_resolution_clock::now();
        if (DO_PARALLEL == "parallel") {
            parallel_cut_voxel(pool, surf_map, pl_fulls, x_buf, win_size);
        }
        else {
            for (int i = 0; i < win_size; i++){
                cut_voxel(surf_map, *pl_fulls[i], x_buf[i], i);
            }
        }

        intime02 = std::chrono::high_resolution_clock::now();

        pcl::PointCloud<PointType> pl_send;
        pub_pl_func(pl_send);

        // pcl::PointCloud<PointType> pl_cent;
        pl_send.clear();
        intime03 = std::chrono::high_resolution_clock::now();

        VOX_HESS voxhess;
        std::cout << "surf_map size: " << surf_map.size() << std::endl;

        for (auto iter = surf_map.begin(); iter != surf_map.end(); iter++)
        {
            if (iter->second == nullptr) {
                std::cerr << "Error: iter->second is null." << std::endl;
                continue;
            }

            try {
                iter->second->recut(win_size);
            }
            catch (const std::exception& e) {
                std::cerr << "Exception during recut: " << e.what() << std::endl;
                continue;
            }

            iter->second->tras_opt(voxhess, win_size);
            iter->second->tras_display(pl_send, win_size);
        }

        /*
        for (auto iter = surf_map.begin(); iter != surf_map.end(); iter++)
        {   
            if (iter->second == nullptr) {
                std::cerr << "Error: iter->second is null." << std::endl;
                continue;
            }
            iter->second->recut(win_size);
            iter->second->tras_opt(voxhess, win_size);
            iter->second->tras_display(pl_send, win_size);
        }
        */

        intime04 = std::chrono::high_resolution_clock::now();

        pub_pl_func(pl_send);
        printf("\nThe planes (point association) cut by adaptive voxelization.\n");
        printf("If the planes are too few, the optimization will be degenerated and fail.\n");
        //printf("If no problem, input '1' to continue or '0' to exit...\n");
        // int a; cin >> a; if(a==0) exit(0);
        pl_send.clear(); pub_pl_func(pl_send);

        printf("plvec_voxels.size() = %d\n", voxhess.plvec_voxels.size());
        printf("x_buf.size() = %d\n", x_buf.size());
        if (voxhess.plvec_voxels.size() < 3 * x_buf.size())
        {
            printf("Initial error too large.\n");
            printf("Please loose plane determination criteria for more planes.\n");
            printf("The optimization is terminated.\n");
            exit(0);
        }
        intime05 = std::chrono::high_resolution_clock::now();

        BALM2 opt_lsv;
        opt_lsv.damping_iter(x_buf, voxhess, pool, DO_PARALLEL);
        intime06 = std::chrono::high_resolution_clock::now();

        for (auto iter = surf_map.begin(); iter != surf_map.end();)
        {
            delete iter->second;
            surf_map.erase(iter++);
        }
        surf_map.clear();

        optimize_memory(); //malloc_trim(0);
        intime07 = std::chrono::high_resolution_clock::now();

        // average intime and save it
        avg_intime01 += std::chrono::duration<double>(intime02 - intime01);
        avg_intime02 += std::chrono::duration<double>(intime03 - intime02);
        avg_intime03 += std::chrono::duration<double>(intime04 - intime03);
        avg_intime04 += std::chrono::duration<double>(intime05 - intime04);
        avg_intime05 += std::chrono::duration<double>(intime06 - intime05);
        avg_intime06 += std::chrono::duration<double>(intime07 - intime06);
    }
    auto time05 = std::chrono::high_resolution_clock::now();

    printf("\nRefined point cloud is publishing...\n");
    //optimize_memory(); //malloc_trim(0);
    createDirectory(SAVE_DIR + "\\after");
    data_show(x_buf, pl_fulls, SAVE_DIR + "\\after");
    printf("\nRefined point cloud is published.\n");
    auto time06 = std::chrono::high_resolution_clock::now();

    printf("\nTime consumption:\n");
    printf("Read file: %f s\n", std::chrono::duration<double>(time02 - time01).count());
    printf("Data transpose: %f s\n", std::chrono::duration<double>(time03 - time02).count());
    printf("Data show: %f s\n", std::chrono::duration<double>(time04 - time03).count());
    printf("Optimization: %f s\n", std::chrono::duration<double>(time05 - time041).count());
    printf("Data show: %f s\n", std::chrono::duration<double>(time06 - time05).count());
    printf("\nAverage time consumption:\n");
    printf("Cut voxel: %f s\n", avg_intime01.count() / 1);
    printf("publish 1: %f s\n", avg_intime02.count() / 1);
    printf("octree calculation: %f s\n", avg_intime03.count() / 1);
    printf("publish 2: %f s\n", avg_intime04.count() / 1);
    printf("damping iter: %f s\n", avg_intime05.count() / 1);
    printf("clean: %f s\n", avg_intime06.count() / 1);

    cout << "============================================================================================" << endl;
    cout << "==================================== !!!!! FINISH !!!!! ====================================" << endl;
    cout << "============================================================================================" << endl;


    return 0;

}


