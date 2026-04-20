"""
可视化生成的动作
用法: python visualize_motion.py --motion test_walk.npy --output test_walk.mp4 --text "a person walks forward"
"""
import numpy as np
import argparse
from utils.plot_script import plot_3d_motion

# HumanML3D kinematic tree
t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]

def recover_from_ric(data, joints_num=22):
    """从263维恢复到3D关节位置"""
    # data shape: (T, 263)
    # 263 = 22*3 (positions) + 22*6 (rotations) + 4 (root) + 2 (foot contact) + 1 (velocity)
    # 简化版本：只取前22*3=66维的位置信息
    positions = data[:, :joints_num*3].reshape(-1, joints_num, 3)
    return positions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--motion', type=str, required=True, help='Motion .npy file')
    parser.add_argument('--output', type=str, default='motion.mp4', help='Output video file')
    parser.add_argument('--text', type=str, default='Generated Motion', help='Title text')
    parser.add_argument('--fps', type=int, default=20, help='FPS for video')
    args = parser.parse_args()

    print(f"Loading motion from: {args.motion}")
    motion_data = np.load(args.motion)
    print(f"Motion shape: {motion_data.shape}")

    # 恢复3D关节位置
    joints_3d = recover_from_ric(motion_data)
    print(f"3D joints shape: {joints_3d.shape}")

    print(f"\nGenerating video: {args.output}")
    print(f"Title: {args.text}")
    print("This may take a few minutes...")

    # 生成视频
    plot_3d_motion(
        args.output,
        t2m_kinematic_chain,
        [joints_3d],
        args.text,
        fps=args.fps
    )

    print(f"\nVideo saved to: {args.output}")

if __name__ == '__main__':
    main()
