import numpy as np

from yahtzee import YahtzeeEnv  # YahtzeeEnv 클래스가 yahtzee_env.py 파일에 있다고 가정합니다.


def test_yahtzee_env():
    env = YahtzeeEnv()
    
    # 환경 초기화 테스트
    obs = env.reset()
    assert len(obs) == 19, f"Observation length should be 20, but got {len(obs)}"
    assert env.rolls_left == 2, f"Rolls left should be 2 after reset, but got {env.rolls_left}"
    
    print("Environment initialization test passed.")

    # 주사위 굴리기 테스트
    env._roll_dice()
    assert all(1 <= d <= 6 for d in env.dice), f"All dice should be between 1 and 6, but got {env.dice}"
    
    print("Dice rolling test passed.")

    # 유효한 액션 테스트
    valid_actions = env._get_valid_actions()
    assert all(0 <= a < 44 for a in valid_actions), f"All actions should be between 0 and 43, but got {valid_actions}"
    
    print("Valid actions test passed.")

    # 게임 플레이 시뮬레이션
    total_reward = 0
    for _ in range(13):  # 13 rounds in Yahtzee
        done = False
        while not done:
            action = np.random.choice(env._get_valid_actions())
            obs, reward, done, info = env.step(action)
            total_reward += reward
                    
        env.reset()
    
    print(f"Game simulation completed. Total reward: {total_reward}")

    # 점수 계산 테스트
    env.dice = np.array([1, 2, 3, 4, 5])
    assert env._calculate_score(9) == 30, "Small straight score calculation failed"
    assert env._calculate_score(10) == 40, "Large straight score calculation failed"
    
    env.dice = np.array([2, 2, 2, 3, 3])
    assert env._calculate_score(8) == 25, "Full house score calculation failed"
    
    print("Score calculation test passed.")

    print("All tests passed successfully!")

if __name__ == "__main__":
    test_yahtzee_env()