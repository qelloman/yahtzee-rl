import numpy as np
import torch
import torch.nn.functional as F


def eval(agent, env, eval_num, device, log_dir):

    with open(f'{log_dir}/eval_log.txt', 'a') as f:
        f.write("=" * 30 + "\n")
        f.write("평가 시작\n")
        
        all_episode_rewards = []
        for eval_idx in range(eval_num):
            f.write(f"평가 {eval_idx}번\n")
            f.write("-" * 30 + "\n")
            episode_rewards = 0
            obs, info = env.reset()
            done = False
            while not done:
                # 주사위 값 복원
                dice_values = []
                for i in range(5):
                    dice_value = np.argmax(obs[i*6:(i+1)*6]) + 1
                    dice_values.append(dice_value)
                
                # 남은 굴림 횟수 복원
                rolls_left = np.argmax(obs[-3:])
                
                f.write(f"dice, rolls_left: : {dice_values}, {rolls_left}\n")
                f.write(f"action mask: {info['action_mask']}\n")
                with torch.no_grad():
                    logits = agent.actor(torch.Tensor(obs).unsqueeze(0).to(device))
                    action_probs = F.softmax(logits.squeeze(), dim=-1)
                    
                    f.write(f"hold probs: {action_probs[:31]}\n")
                    f.write(f"action probs: {action_probs[31:]}\n")
                    masked_action_prob_sum = 0.0
                    for idx, mask in enumerate(info['action_mask']):
                        if not mask:
                            masked_action_prob_sum += action_probs[idx]
                    
                    action, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).unsqueeze(0).to(device))
                    action_txt = 'wrong action' if not info['action_mask'][action] else 'correct action'
                    f.write(f"{action_txt}, masked action prob sum = {masked_action_prob_sum}\n")
                obs, reward, done, trunc, info = env.step(action.item())
                f.write(f"action, reward, done: {action}, {reward}, {done}\n")
                episode_rewards += reward
            all_episode_rewards.append(episode_rewards)
            f.write(f"total_reward: {episode_rewards}\n")
        f.write("=" * 30 + "\n\n")
        return np.mean(all_episode_rewards)