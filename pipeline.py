import trainTestDataShuffle
import os
import subprocess
import sys
# Execute a simple command

def run_command(command, env_vars=None):
    """
    執行一個命令，同時設置指定的環境變數。

    Parameters:
    - command (list): 要執行的命令及其參數。
    - env_vars (dict, optional): 要設置的環境變數。

    Returns:
    - None
    """
    # 複製當前的環境變數
    env = os.environ.copy()
    
    # 如果有新的環境變數，將它們加入
    if env_vars:
        env.update(env_vars)
    
    try:
        # 執行命令
        result = subprocess.run(
            command,
            env=env,
            check=True,  # 如果命令返回非零狀態，會引發 CalledProcessError
            capture_output=True,  # 捕捉標準輸出和標準錯誤
            text=True  # 將輸出作為字符串返回
        )
        print(f"命令 `{ ' '.join(command) }` 執行成功。")
        # print("標準輸出:")
        # print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"命令 `{ ' '.join(command) }` 執行失敗，返回碼 {e.returncode}。")
        print("標準錯誤:")
        print(e.stderr)
        sys.exit(e.returncode)  # 根據需要退出腳本或進行其他處理

def main_routine():
    # 定義所有需要執行的步驟
    steps = [
        {
            "env": {"CHANNEL_SELECTED": "0"},
            "command": ["python", "-u", "trainTest_I.py", "-train"],
            "additional_env": {
                "EMBEDDING_SIZE":"8"
            }
        },
        {
            "env": {"PHASEII_MODE": "amp"},
            "command": ["python", "-u", "trainTest_II.py", "-train", "-test"],
            "additional_env": {
                "EMBEDDING_SIZE":"8",
                "PHASEII_TRAIN_SET_PATH": "D:/leveling/pytorch-AE/trainTest_II_dataSets/train_set_amp.joblib",
                "PHASEII_TEST_SET_PATH": "D:/leveling/pytorch-AE/trainTest_II_dataSets/test_set_amp.joblib"
            }
        },
        {
            "env": {"CHANNEL_SELECTED": "1 2 3"},
            "command": ["python", "-u", "trainTest_I.py", "-train"],
            "additional_env": {
                "EMBEDDING_SIZE": "16"
            }
        },
        {
            "env": {"PHASEII_MODE": "vib"},
            "command": ["python", "-u", "trainTest_II.py", "-train", "-test"],
            "additional_env": {
                "EMBEDDING_SIZE": "16",
                "CHANNEL_SELECTED": "1 2 3",
                "PHASEII_TRAIN_SET_PATH": "D:/leveling/pytorch-AE/trainTest_II_dataSets/train_set_vib.joblib",
                "PHASEII_TEST_SET_PATH": "D:/leveling/pytorch-AE/trainTest_II_dataSets/test_set_vib.joblib"
            }
        },
        {
            "env": None,  # 最後一步不需要設置環境變數
            "command": ["python", "mix.py"]
        }
    ]

    for idx, step in enumerate(steps, 1):
        print(f"\n執行步驟 {idx}: {' '.join(step['command'])}")
        
        # 準備環境變數
        env_vars = step.get("env", {})
        additional_env = step.get("additional_env", {})
        if additional_env:
            env_vars.update(additional_env)
        
        # 執行命令
        run_command(step["command"], env_vars=env_vars)
        
routines = 4

for i in range(routines):
    trainTestDataShuffle.main()
    main_routine()
# train.main()
# test.main()