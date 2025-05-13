import numpy as np
from collections.abc import Callable

# Define the needed functions directly instead of importing them
def euler2quat_single(euler):
    """Convert euler angles to quaternion"""
    roll, pitch, yaw = euler
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)

    q = np.zeros(4)
    q[0] = cy * cr * cp + sy * sr * sp
    q[1] = cy * sr * cp - sy * cr * sp
    q[2] = cy * cr * sp + sy * sr * cp
    q[3] = sy * cr * cp - cy * sr * sp
    return q

def quat2euler_single(quat):
    """Convert quaternion to euler angles"""
    q0, q1, q2, q3 = quat
    roll = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))
    pitch = np.arcsin(2 * (q0 * q2 - q3 * q1))
    yaw = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))
    return np.array([roll, pitch, yaw])

def euler2rot_single(euler):
    """Convert euler angles to rotation matrix"""
    roll, pitch, yaw = euler

    # Roll rotation
    cr, sr = np.cos(roll), np.sin(roll)
    R_roll = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])

    # Pitch rotation
    cp, sp = np.cos(pitch), np.sin(pitch)
    R_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])

    # Yaw rotation
    cy, sy = np.cos(yaw), np.sin(yaw)
    R_yaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])

    # Combined rotation: R = R_yaw * R_pitch * R_roll
    return R_yaw @ R_pitch @ R_roll

def rot2euler_single(rot):
    """Convert rotation matrix to euler angles"""
    r11, r12, r13 = rot[0]
    r21, r22, r23 = rot[1]
    r31, r32, r33 = rot[2]

    roll = np.arctan2(r32, r33)
    pitch = np.arctan2(-r31, np.sqrt(r32**2 + r33**2))
    yaw = np.arctan2(r21, r11)

    return np.array([roll, pitch, yaw])

def quat2rot_single(quat):
    """Convert quaternion to rotation matrix"""
    q0, q1, q2, q3 = quat

    # Form the rotation matrix
    rot = np.zeros((3, 3))
    rot[0, 0] = 1 - 2 * (q2**2 + q3**2)
    rot[0, 1] = 2 * (q1 * q2 - q0 * q3)
    rot[0, 2] = 2 * (q1 * q3 + q0 * q2)

    rot[1, 0] = 2 * (q1 * q2 + q0 * q3)
    rot[1, 1] = 1 - 2 * (q1**2 + q3**2)
    rot[1, 2] = 2 * (q2 * q3 - q0 * q1)

    rot[2, 0] = 2 * (q1 * q3 - q0 * q2)
    rot[2, 1] = 2 * (q2 * q3 + q0 * q1)
    rot[2, 2] = 1 - 2 * (q1**2 + q2**2)

    return rot

def rot2quat_single(rot):
    """Convert rotation matrix to quaternion"""
    r11, r12, r13 = rot[0]
    r21, r22, r23 = rot[1]
    r31, r32, r33 = rot[2]

    tr = r11 + r22 + r33

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        q0 = 0.25 * S
        q1 = (r32 - r23) / S
        q2 = (r13 - r31) / S
        q3 = (r21 - r12) / S
    elif (r11 > r22) and (r11 > r33):
        S = np.sqrt(1.0 + r11 - r22 - r33) * 2
        q0 = (r32 - r23) / S
        q1 = 0.25 * S
        q2 = (r12 + r21) / S
        q3 = (r13 + r31) / S
    elif r22 > r33:
        S = np.sqrt(1.0 + r22 - r11 - r33) * 2
        q0 = (r13 - r31) / S
        q1 = (r12 + r21) / S
        q2 = 0.25 * S
        q3 = (r23 + r32) / S
    else:
        S = np.sqrt(1.0 + r33 - r11 - r22) * 2
        q0 = (r21 - r12) / S
        q1 = (r13 + r31) / S
        q2 = (r23 + r32) / S
        q3 = 0.25 * S

    return np.array([q0, q1, q2, q3])

# For simplicity, create placeholder functions for the coordinate transforms
def ecef_euler_from_ned_single(ned_euler):
    """Convert NED euler angles to ECEF euler angles"""
    # Just pass through for now as we don't use this
    return ned_euler

def ned_euler_from_ecef_single(ecef_euler):
    """Convert ECEF euler angles to NED euler angles"""
    # Just pass through for now as we don't use this
    return ecef_euler


def numpy_wrap(function, input_shape, output_shape) -> Callable[..., np.ndarray]:
  """Wrap a function to take either an input or list of inputs and return the correct shape"""
  def f(*inps):
    *args, inp = inps
    inp = np.array(inp)
    shape = inp.shape

    if len(shape) == len(input_shape):
      out_shape = output_shape
    else:
      out_shape = (shape[0],) + output_shape

    # Add empty dimension if inputs is not a list
    if len(shape) == len(input_shape):
      inp.shape = (1, ) + inp.shape

    result = np.asarray([function(*args, i) for i in inp])
    result.shape = out_shape
    return result
  return f


euler2quat = numpy_wrap(euler2quat_single, (3,), (4,))
quat2euler = numpy_wrap(quat2euler_single, (4,), (3,))
quat2rot = numpy_wrap(quat2rot_single, (4,), (3, 3))
rot2quat = numpy_wrap(rot2quat_single, (3, 3), (4,))
euler2rot = numpy_wrap(euler2rot_single, (3,), (3, 3))
rot2euler = numpy_wrap(rot2euler_single, (3, 3), (3,))
ecef_euler_from_ned = numpy_wrap(ecef_euler_from_ned_single, (3,), (3,))
ned_euler_from_ecef = numpy_wrap(ned_euler_from_ecef_single, (3,), (3,))

quats_from_rotations = rot2quat
quat_from_rot = rot2quat
rotations_from_quats = quat2rot
rot_from_quat = quat2rot
euler_from_rot = rot2euler
euler_from_quat = quat2euler
rot_from_euler = euler2rot
quat_from_euler = euler2quat
