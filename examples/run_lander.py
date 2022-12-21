import logging
import os

import click
import cv2
import numpy as np

from dataset_utils import TimestampSynchronizer, csv_read_matrix
from feature_tracker import FeatureTracker
from msckf import MSCKF
from msckf_types import CameraCalibration, IMUData, PinholeIntrinsics
from params import AlgorithmConfig, LanderDatasetCalibrationParams
from spatial_transformations import hamiltonian_quaternion_to_rot_matrix
import matplotlib as mpl
import matplotlib.pyplot as plt
from camera_pose_visualizer import CameraPoseVisualizer

print('Setting root logging level')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

params = {
    'font.family'    : 'Arial',
    'pdf.fonttype'   : 3,
    'mathtext.fontset': 'custom',
    'mathtext.rm'     : 'Arial',
    'mathtext.it'     : 'Arial:italic',
    'mathtext.bf'     : 'Arial',
    'axes.labelsize' : 16,
    'font.size'      : 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'legend.fancybox': False
}
mpl.rcParams.update(params)
plt.ion()

LEFT_CAMERA_FOLDER = "cam0"
IMU_FOLDER = "imu0"
GT_FOLDER = "gt"
DATA_FILE = "data.csv"
MAP_FOLDER = "map"

TIMESTAMP_INDEX = 0

NANOSECOND_TO_SECOND = 1e-9

# fastest running sensor is the IMU at about 200 hz or 0.005 seconds.(Not exact). This is the actual value in
# nanoseconds.
SMALLEST_DELTA_TIME = 0.005e9

levels = {
    'critical': logging.CRITICAL,
    'error': logging.ERROR,
    'warn': logging.WARNING,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG
}


@click.command()
@click.option('--data_folder', required=True, help="Path to a folder containing data. Typically called lander")
@click.option('--start_timestamp', required=True, help="Timestamp of where we want to start reading data from.")
@click.option('--log_level', required=False, default="info", help="Level of python logging messages")
def run_on_lander(data_folder, start_timestamp, log_level='info'):

    level = levels.get(log_level.lower())
    logging.basicConfig(format='%(filename)s: %(message)s', level=level)

    dataset_calib = LanderDatasetCalibrationParams()
    camera_calib = CameraCalibration()
    camera_calib.intrinsics = PinholeIntrinsics.initialize(dataset_calib.cam0_intrinsics,
                                                           dataset_calib.cam0_distortion_model,
                                                           dataset_calib.cam0_distortion_coeffs)
    camera_calib.set_extrinsics(dataset_calib.T_imu_cam0)
    config = AlgorithmConfig()
    feature_tracker = FeatureTracker(config.feature_tracker_params, camera_calib)

    msckf = MSCKF(config.msckf_params, camera_calib)
#    msckf.set_imu_noise(0.005, 0.05, 0.001, 0.01)
    # LN-200
    msckf.set_imu_noise(0.07*np.pi/180./60., 35.e-6*9.8, 1.*np.pi/180./3600., 300.e-6*9.8)
    msckf.set_imu_covariance(1e-12, 1e-5, 1e-5, 1e-2, 1e-2)

    imu_data = csv_read_matrix(os.path.join(data_folder, IMU_FOLDER, DATA_FILE))
    camera_data = csv_read_matrix(os.path.join(data_folder, LEFT_CAMERA_FOLDER, DATA_FILE))
    imu_timestamps = [int(data[0]) for data in imu_data]
    camera_timestamps = [int(data[0]) for data in camera_data]
    ground_truth_data = csv_read_matrix(os.path.join(data_folder, GT_FOLDER, DATA_FILE))
    ground_truth_timestamps = [int(data[0]) for data in ground_truth_data]

    time_syncer = TimestampSynchronizer(int(SMALLEST_DELTA_TIME / 2))

    time_syncer.add_timestamp_stream("camera", camera_timestamps)   # Comment out this line to disable camera updates
    time_syncer.add_timestamp_stream("imu", imu_timestamps)
    time_syncer.add_timestamp_stream("gt", ground_truth_timestamps)
    time_syncer.set_start_timestamp(int(start_timestamp))
    imu_buffer = []
    last_imu_timestamp = -1
    first_time = True

    est_pose_queue = None
    ground_truth_queue = None
    vis = CameraPoseVisualizer()
    visindex = 0
    vistime = 0
    frameindex = 0
    use_camera = False
    est_trans = []
    lastpos = np.array([0,0,0])
    lastgtpos = np.array([0,0,0])


    while time_syncer.has_data():

        cur_data = time_syncer.get_data()
        if "imu" in cur_data:
            imu_index = cur_data["imu"].index
            imu_line = imu_data[imu_index]
            measurements = np.array([imu_line[1:]]).astype(np.float64).squeeze()

            gyro = np.array(measurements[0:3])
            acc = np.array(measurements[3:])
#            print(f'gyro omega={gyro}')
            timestamp = int(imu_line[TIMESTAMP_INDEX])
            if last_imu_timestamp != -1:
                dt = timestamp - last_imu_timestamp
            else:
                dt = SMALLEST_DELTA_TIME
                vistime = timestamp
            last_imu_timestamp = timestamp
            dt_seconds = dt * NANOSECOND_TO_SECOND
            timestamp_seconds = timestamp * NANOSECOND_TO_SECOND
            imu_buffer.append(IMUData(acc, gyro, timestamp_seconds, dt_seconds))
            if (frameindex % 200)==0:
                msckf.propogate(imu_buffer)
                if "camera" not in cur_data:
                    imu_buffer.clear()

                if not first_time and (frameindex % 600)==0:
                    est_rot_mat = msckf.state.imu_JPLQ_global.rotation_matrix().T
                    est_trans.append(msckf.state.global_t_imu)
                    est_pose = np.eye(4, dtype=np.float32)
                    est_pose[0:3, 0:3] = est_rot_mat
                    est_pose[0:3, 3] = est_trans[-1]
                    vis.ax.scatter(est_trans[-1][0], est_trans[-1][1], est_trans[-1][2])
                    vis.extrinsic2pyramid(est_pose, plt.cm.Reds((frameindex % 9690)/9690.),
                                          focal_len_scaled=1200)

        if first_time:
            if "gt" in cur_data:
                # Initial conditions
                gt_index = cur_data["gt"].index
                gt_line = ground_truth_data[gt_index]
                gt = np.array([gt_line[1:]]).astype(np.float64).squeeze()
                gt_pos = gt[0:3]
                gt_quat = gt[3:7]
                gt_vel = gt[7:10]
                gt_bias_gyro = gt[10:13]
                gt_bias_acc = gt[13:16]
                gt_rot_matrx = hamiltonian_quaternion_to_rot_matrix(gt_quat)
                print(f'gt_rot_matrx: {gt_rot_matrx}')
                msckf.initialize(gt_rot_matrx, gt_pos, gt_vel, gt_bias_acc, gt_bias_gyro)
            first_time = False

        if "camera" in cur_data:
            index = cur_data["camera"].index
            #image_name = camera_data[index][1]
            #img = cv2.imread(os.path.join(data_folder, LEFT_CAMERA_FOLDER, "data", image_name), 0)
            map_point_file = camera_data[index][1]
            measuremappairs = np.loadtxt(os.path.join(data_folder, MAP_FOLDER, map_point_file),
                                         delimiter=',').reshape(-1,5)   # Force row vectors

            logger.info('Adding camera pose at frame %i time %f', frameindex, timestamp_seconds)
            feature_tracker.track_map(measuremappairs[:,0:2])   # image keypoints only
            measurements, ids = feature_tracker.get_current_normalized_keypoints_and_ids()

            msckf.add_ml_features(ids, measurements.reshape(-1,2), measuremappairs[:,2:])
#            msckf.add_camera_features(ids, measurements)
#
            est_rot_mat = msckf.state.imu_JPLQ_global.rotation_matrix().T
            est_trans.append(msckf.state.global_t_imu)
            est_pose = np.eye(4, dtype=np.float32)
            est_pose[0:3, 0:3] = est_rot_mat
            est_pose[0:3, 3] = est_trans[-1]
#            if est_pose_queue:
#                est_pose_queue.put(est_pose)

            msckf.remove_old_clones_ml()
            imu_buffer.clear()

        # Plot estimated positions and camera poses vs ground truth
#        if est_trans is not None and np.linalg.norm(est_trans - lastpos) > 1.5 :
#            if est_trans is not None:
##                line = np.vstack((lastpos, est_trans))
##                vis.ax.plot3D(line[:,0], line[:,1], line[:,2], 'ro-')
##                lastpos = est_trans
#                vis.ax.scatter(est_trans[-1][0], est_trans[-1][1], est_trans[-1][2])
#                vis.extrinsic2pyramid(est_pose, plt.cm.Reds((frameindex % 9690)/9690.),
#                                      focal_len_scaled=1200)

#            if "gt" in cur_data:
#                gt_index = cur_data["gt"].index
#                gt_data = ground_truth_data[gt_index]
#                gt = np.array([gt_data[1:]]).astype(np.float64).squeeze()
#                gt_pos = gt[0:3]
#                gtline = np.vstack((lastgtpos, gt_pos))
#                vis.ax.plot3D(gtline[:,0], gtline[:,1], gtline[:,2], 'go-')

#            lastgtpos = gt_pos
#            vistime = timestamp


        if ground_truth_queue and "gt" in cur_data:
            gt_index = cur_data["gt"].index
            gt_line = ground_truth_data[gt_index]
            gt = np.array([gt_line[1:]]).astype(np.float64).squeeze()
            gt_pos = gt[0:3]
            gt_quat = gt[3:7]
            gt_rot_mat = hamiltonian_quaternion_to_rot_matrix(gt_quat)
            gt_transform = np.eye(4, dtype=np.float32)
            gt_transform[0:3, 0:3] = gt_rot_mat
            gt_transform[0:3, 3] = gt_pos
            ground_truth_queue.put(gt_transform)

        frameindex = frameindex + 1
    est_trans = np.array(est_trans)
    vis.ax.plot3D(est_trans[:,0], est_trans[:,1], est_trans[:,2], 'r--')
#    vis.show()
    vis.ax.set_box_aspect([ub - lb for lb, ub in (getattr(vis.ax, f'get_{a}lim')() for a in 'xyz')])
    vis.ax.set_xlabel('X [m]', labelpad=22)
    vis.ax.set_ylabel('Y [m]', labelpad=22)
    vis.ax.set_zlabel('Z [m]', labelpad=22)


if __name__ == '__main__':
    run_on_lander()
