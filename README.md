# Droid-Semantic-Mapping

This project combines the `Droid-SLAM` and `SemanticBKI` mapping to achieve the monocular camera SLAM and 3D semantic mapping. 

# Requirement

- Operating System: Ubuntu 20.04
- Robot Operating System (ROS)
- GPU with a least 11G of memory

# Getting Started

1. Clone the repo

2. Follow the instructions to install [Droid-SLAM](https://github.com/princeton-vl/DROID-SLAM)

3. Build `bki_ws` using catkin

   ```bash
   bki_ws$ catkin_make
   bki_ws$ source devel/setup.bash
   ```

   

4. Go into `DROID-SLAM` folder and run the demo with user-specified path to video

   See demo.py details in [Droid-SLAM](https://github.com/princeton-vl/DROID-SLAM)

   ```bash
   DROID_SLAM$ python demo.py --imagedir=data/sfm_bench/rgb --calib=calib/eth.txt
   ```

5. Run `MMSegmentation.ipynb ` to get semantic segmentation result

6. Put semantics.py from step 5 in the reconstruction folder

7. Run `io_1.py` under ROB_530 folder to transfer the result from DROID-SLAM to `SemanticKitti` dataset

8. In `bki_ws` run following

   ```bash
   $ mkdir data
   $ cd data
   ```

9. Move `SemanticKitti` dataset to the data folder

10. Get the result by running:

    ```bash
    roslaunch semantic_bki droid.launch
    ```

    



