import argparse


parser = argparse.ArgumentParser()

# 一般設定
parser.add_argument("--algorithm", type = str, default = "CEM-RL")
parser.add_argument("--output_path", type = str, default = "saved", help = "輸出檔案路徑")
parser.add_argument("--save_result", action = "store_true", help = "是否需要紀錄結果")

# 實驗設定
parser.add_argument("--env_name", type = str, default = "HalfCheetah-v5", help = "實驗環境")
parser.add_argument("--device", type = str, default = "cuda:0", help = "實驗使用的設備")
parser.add_argument("--seed", type = int, default = 0, help = "亂數種子")
parser.add_argument("--start_steps", type = int, default = 10000, help = "最開始使用隨機 action 進行探索")
parser.add_argument("--max_steps", type = int, default = int(1e6), help = "最多多少 steps")

# 性能測試設定
parser.add_argument("--test_performance_freq", type = int, default = 1000, help = "每多少 steps 要測試 actor 的性能")
parser.add_argument("--test_n", type = int, default = 20, help = "每次測試 actor 要玩幾局")

# RL設定
parser.add_argument("--replay_buffer_size", type = int, default = int(1e6), help = "Replay buffer 的最大空間")
parser.add_argument("--batch_size", type = int, default = 256, help = "Random mini-batch size")
parser.add_argument("--gamma", type = float, default = 0.99, help = "TD 的 discount")
parser.add_argument("--tau", type = float, default = 0.005, help = "以移動平均更新target的比例")
parser.add_argument("--actor_learning_rate", type = float, default = 3e-4, help = "Actor 的學習率")
parser.add_argument("--critic_learning_rate", type = float, default = 3e-4, help = "Critic 的學習率")

# TD3設定
parser.add_argument('--policy_noise', type = float, default = 0.2)
parser.add_argument('--noise_clip', type = float, default = 0.5)
parser.add_argument('--policy_freq', type = int, default = 2)

# EA設定
parser.add_argument("--population_size", type = int, default = 10)

# CEM設定
parser.add_argument("--CEM_parents_ratio_actor", type = float, default = 0.5 , help = "parents的比例")
parser.add_argument("--CEM_cov_discount_actor", type = float, default = 0.2 , help = "cov折扣")
parser.add_argument("--CEM_sigma_init", type = float, default = 1e-3, help = "CEM一開始的cov")
parser.add_argument('--n_grad', default=5, type=int, help = "訓練一半")

args = parser.parse_args()