#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import argparse
import hashlib
import os
import tempfile
import numpy as np
from pyquaternion import Quaternion

from nuplan_scripts.utils.colmap_utils.database import COLMAPDatabase


def get_colmap_database_path(colmap_path):
    """Return a local SQLite DB path for COLMAP to avoid network/overlay FS locks."""
    db_root = os.environ.get(
        'MTGS_COLMAP_DB_ROOT',
        os.path.join(tempfile.gettempdir(), 'mtgs_colmap_databases')
    )
    os.makedirs(db_root, exist_ok=True)
    abs_path = os.path.abspath(colmap_path)
    digest = hashlib.sha1(abs_path.encode('utf-8')).hexdigest()[:16]
    basename = os.path.basename(abs_path.rstrip(os.sep))
    return os.path.join(db_root, f'{basename}-{digest}.db')


def _remove_sqlite_files(database_path):
    for suffix in ('', '-journal', '-wal', '-shm'):
        path = database_path + suffix
        if os.path.exists(path):
            os.remove(path)


def cleanup_colmap_database(colmap_path):
    _remove_sqlite_files(get_colmap_database_path(colmap_path))


def create_colmap_database(colmap_path, input_model='sparse_model', camera_type='OPENCV'):
    camera_type = 4 if camera_type == 'OPENCV' else 1

    database_path = get_colmap_database_path(colmap_path)
    _remove_sqlite_files(database_path)
    colmap_db = COLMAPDatabase.connect(database_path)
    colmap_db.create_tables()
    for cameras in open(f"{colmap_path}/{input_model}/cameras.txt", "r").readlines():
        cameras = cameras.strip().split()
        colmap_db.add_camera(
            camera_id=int(cameras[0]),
            model=camera_type, # 1 for PINHOLE, 4 for OPENCV
            width=int(cameras[2]),
            height=int(cameras[3]),
            params=cameras[4:],
            prior_focal_length=True
        )
    for image in open(f"{colmap_path}/{input_model}/images.txt", "r").readlines()[::2]:
        image = image.strip().split()
        colmap_db.add_image(
            name=image[-1],
            camera_id=int(image[-2]),
            image_id=int(image[0])
        )

        prior_position = -Quaternion(*image[1:5]).rotation_matrix.T @ np.array(image[5:8], dtype=np.float64)
        colmap_db.add_pose_prior(
            image_id=int(image[0]),
            position=prior_position,
            coordinate_system=1  # 1 for CARTESIAN
        )

    colmap_db.commit()
    colmap_db.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--colmap_path', type=str, help='Input colmap dir', required=True)
    parser.add_argument('--input_model', type=str, default='sparse_model')
    args = parser.parse_args()

    create_colmap_database(args.colmap_path, args.input_model)
