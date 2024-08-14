<p align="center">
  <b>BALM window version</b>
</p>

We modified BALM's code to run on **Windows** and improved time performance by **30 percent** through additional optimizations (additional parallel operations). 😁

### Reference paper: [Efficient and Consistent Bundle Adjustment on Lidar Point Clouds](https://arxiv.org/pdf/2209.08854)
### Reference code: [BALM 2.0](https://github.com/hku-mars/BALM)

We made the code run in a Windows Visual Studio environment. You can use any version of visual studio **19** or **22** you want. 😁


## Prerequisited

### 1.1 Windows 10/11 and Visual studio 19/22

check your windows version

```
# To check your Windows version, open Command Prompt (cmd) and type:
winver
```

Install [Visual Studio](https://visualstudio.microsoft.com/ko/downloads/)

### 1.2 PCL and Eigen

Follow [PCL Installation](https://pointclouds.org/) (1.10 recommended)

Follow [Eigen Installation](https://eigen.tuxfamily.org/index.php?title=Main_Page) (3.3.7 recommended)

You can easily download those using **vcpkg.**

```
# open Command Prompt (cmd) and type:
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.bootstrap-vcpkg.bat
```

**You need to register vcpkg.exe in environment variables.**

If you download vcpkg and add it to the environment variables,
```
# open Command Prompt (cmd) and type:
vcpkg integrate install
vcpkg install pcl
vcpkg install eigen3
```


## Environments 
You can configure the environment by running **.vcproj** in the folder through visual studio. **(Please check the environment variables carefully.)**

## Run winBALM
Before running the winBALM, please set **the path (BASE_DIR, SAVE_DIR, SCENE_NAME)** properly in **hyperparameters.yaml.**

If you want **to reduce the frame**, you can use **limit frame.**

For hyperparameters other than **voxel_size, eigen value array,** etc., **refer to the top of ba_voxel.hpp** for hyperparameters you want to add.
