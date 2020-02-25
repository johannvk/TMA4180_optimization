# File defining functions for visualizing robot-arm positions.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines


def get_robot_line_segments(line_lengths, turn_angles):
    assert(len(line_lengths) == len(turn_angles))

    x_coordinates = [0.0]
    y_coordinates = [0.0]

    current_angle = 0.0

    for i, l in enumerate(line_lengths):
        current_angle += turn_angles[i]
        x_coordinates.append(x_coordinates[i] + l * np.cos(current_angle))
        y_coordinates.append(y_coordinates[i] + l * np.sin(current_angle))

    return x_coordinates, y_coordinates


def display_robot_arm(line_lengths, turn_angles, target_point=False, display=True, filepath=False):
    line_x, line_y = get_robot_line_segments(line_lengths, turn_angles)

    robot_arm = lines.Line2D(line_x, line_y,
                             c='green', alpha=0.8, linestyle='-', linewidth=2, marker='.', markersize=10, markerfacecolor='red')

    fig, axis = plt.subplots()
    axis.add_line(robot_arm)
    axis.plot(0, 0, "bo")  # Plotting Origin
    end_x, end_y = line_x[-1], line_y[-1]
    axis.plot(end_x, end_y, "ko")  # Plotting end point of the arm:

    if target_point:
        axis.plot(target_point[0], target_point[1], 'x')

    fig.suptitle("Robot Arm", fontsize=18)
    plt.tight_layout()

    axis.autoscale()
    axis.set(aspect=1)

    if display:
        plt.show()
    else:
        plt.savefig(filepath, bbox_inces='tight')


def test_display_robot_arm():
    line_length = [1.0, 2.0, 3.0, 4.0, 5.0]
    angles = [np.pi/3, -np.pi/3, -np.pi/3, np.pi/3, -np.pi/3]

    display_robot_arm(line_length, angles)


if __name__ == "__main__":
    test_display_robot_arm()
