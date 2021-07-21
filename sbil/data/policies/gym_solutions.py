# https://github.com/ZhiqingXiao/OpenAIGymSolution
import gym
import numpy as np

def cartpole(observations):
    position, velocity, angle, angle_velocity, _ = np.split(observations, np.arange(1,5), axis=-1)
    action = (3 * angle + angle_velocity) > 0
    return action.astype(int)[...,0]

def mountain_car(observations):
    position, velocity, _ = np.split(observations, np.arange(1,3), axis=-1)
    lb = np.minimum(-0.09 * (position + 0.25) ** 2 + 0.03,
            0.3 * (position + 0.9) ** 4 - 0.008)
    ub = -0.07 * (position + 0.38) ** 2 + 0.07
    action = (lb < velocity < ub)*2
    return action.astype(int)[...,0]

def mountain_car_continuous(observations):
    position, velocity, _ = np.split(observations, np.arange(1,3), axis=-1)
    return np.logical_or(position > -4 * velocity, position < 13 * velocity - 0.6)*2-1


def pendulum(observations):
    x, y, angle_velocity, _ = np.split(observations, np.arange(1,4), axis=-1)
    flip = (y < 0.)
    y *= -2*flip+1 # change sign
    angle_velocity *= -2*flip+1
    angle = np.arcsin(y)
    angle = np.where(x<0, np.pi - angle, angle)
    force = np.where(
        np.logical_or(
            angle < -0.3 * angle_velocity,
            np.logical_and(
                angle > 0.03 * (angle_velocity - 2.5) ** 2. + 1.,
                angle < 0.15 * (angle_velocity + 3.) ** 2. + 2.
            )
        ), 2, -2
    )
    force *= -2*flip+1
    return force


def acrobot(observations):
    x0, y0, x1, y1, v0, v1, _ = np.split(observations, np.arange(1,7), axis=-1)
    return np.where(
        np.logical_or(v1 < -0.3, np.logical_and(-0.3 < v1 < 0.3, y1 + x0 * y1 + x1 * y0 > 0)),
        0, 2
    ).astype(int)[...,0]


def lunar_lander(observations):
    x, y, v_x, v_y, angle, v_angle, contact_left, contact_right, _ = np.split(observations, np.arange(1,8), axis=-1)

    contact = np.logical_or(contact_left, contact_right)
    f_y = np.where(contact, -10. * v_y - 1., 5.5 * np.abs(x) - 10. * y - 10. * v_y - 1.)
    f_angle = np.where(contact, 0, -np.clip(5. * x + 10. * v_x, -4, 4) + 10. * angle + 20. * v_angle)


    return np.select(
        [
            np.logical_and(np.abs(f_angle) <= 1, f_y <= 0),
            f_angle < f_y + (y<0),
            np.abs(f_angle) < f_y,
            f_angle < -(f_y + (y<0))
        ],
        [0, 1, 2, 3]
    ).astype(int)[...,0]


def lunar_lander_continuous(observations):
    x, y, v_x, v_y, angle, v_angle, contact_left, contact_right, _ = np.split(observations, np.arange(1,8), axis=-1)
    contact = np.logical_or(contact_left, contact_right)
    return np.where(
        contact,
        np.stack([-10. * v_y - 1., 0], axis=-1),
        np.stack(
            [
                5.5 * np.abs(x) - 10. * y - 10. * v_y - 1.,
                -np.clip(5. * x + 10. * v_x, -4, 4) + 10. * angle + 20. * v_angle
            ],
            axis=-1
        )
    )

def bipedal_walker(observations): # v-3
    obs = np.array(observations)
    weights = np.array([
        [ 0.9, -0.7,  0.0, -1.4], [ 4.3, -1.6, -4.4, -2.0], [ 2.4, -4.2, -1.3, -0.1],
        [-3.1, -5.0, -2.0, -3.3], [-0.8,  1.4,  1.7,  0.2], [-0.7,  0.2, -0.2,  0.1],
        [-0.6, -1.5, -0.6,  0.3], [-0.5, -0.3,  0.2,  0.1], [ 0.0, -0.1, -0.1,  0.1],
        [ 0.4,  0.8, -1.6, -0.5], [-0.4,  0.5, -0.3, -0.4], [ 0.3,  2.0,  0.9, -1.6],
        [ 0.0, -0.2,  0.1, -0.3], [ 0.1,  0.2, -0.5, -0.3], [ 0.7,  0.3,  5.1, -2.4],
        [-0.4, -2.3,  0.3, -4.0], [ 0.1, -0.8,  0.3,  2.5], [ 0.4, -0.9, -1.8,  0.3],
        [-3.9, -3.5,  2.8,  0.8], [ 0.4, -2.8,  0.4,  1.4], [-2.2, -2.1, -2.2, -3.2],
        [-2.7, -2.6,  0.3,  0.6], [ 2.0,  2.8,  0.0, -0.9], [-2.2,  0.6,  4.7, -4.6],
    ])
    return obs[...,:24] @ weights + np.array([3.2, 6.1, -4.0, 7.6])
