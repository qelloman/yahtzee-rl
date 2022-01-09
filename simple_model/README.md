# Simple Model #
Instead of playing full yahtzee game, `SimpleYahtzeeEnv` environment emulates simplified version of yahtezee.

- The score categories only consist of ones, twos, and so forth.
- We roll dice once for on score category. (No 3 rolls)
- We can choose the number of dice (`num_dice`)
- We can choose the number of dice's face, which is the number of categories (`num_dice_face`)

NOTE: We can choose an option to give reward for every action or the last action.
- Every action: reward is given for the corresponding score category.
- Last action: reward is given for the total score. Otherwise, reward is zero.

# Agent #
We have three different agents.
1. human agent (`HumanAgent`): we need to input each action
2. base agent (`BaseAgent`): every action is randomly chosen.
3. qlearning agent (`QLearningAgent`): state-action values are learned and action is chosen based on the values (greedy policy.) 

Each agent file has runnable code.

# Human Agent #
```commandline
python human_agent.py
You rolled 3 dice. => [2 4 4]
| category | score |
|     1     |  00 |
|     2     |  00 |
|     3     |  00 |
|     4     |  00 |

Available categories = [1, 2, 3, 4]
Enter category[1-#], q for quit: 4

you picked 4 out of [1, 2, 3, 4]. => score = 8
You rolled 3 dice. => [1 1 2]

| category | score |
|     1     |  00 |
|     2     |  00 |
|     3     |  00 |
|     4     |  08 |

Available categories = [1, 2, 3]
Enter category[1-#], q for quit: 1

you picked 1 out of [1, 2, 3]. => score = 2
You rolled 3 dice. => [1 2 2]

| category | score |
|     1     |  02 |
|     2     |  00 |
|     3     |  00 |
|     4     |  08 |

Available categories = [2, 3]
Enter category[1-#], q for quit: 2

you picked 2 out of [2, 3]. => score = 4
You rolled 3 dice. => [1 3 4]

| category | score |
|     1     |  02 |
|     2     |  04 |
|     3     |  00 |
|     4     |  08 |

Available categories = [3]
Enter category[1-#], q for quit: 3

| category | score |
|     1     |  02 |
|     2     |  04 |
|     3     |  03 |
|     4     |  08 |

==== Total Score: 17 ====
```

# Base agent Vs QLearning Agent #

### How to run ###
``` python compare_multiple_size_yahtzee.py```

### Parameters ###
- 5,000,000 episodes for learning, 1,000 episodes for test
- epsilon start=1.0, epsilon decaying rate = 0.99995, epsilon min=0.3
- alpha = 0.4
- gamma = 1

``` python compare_multiple_size_yahtzee.py```

### Last-action-only reward environment ###
| num_dice | num_dice_face | base (random) | Qlearning |
|----------|---------------|---------------|-----------|
| 3        | 2             | 4.485         | 3.965     |
| 4        | 2             | 5.901         | 6.24      |
| 5        | 2             | 7.462         | 8.109     |
| 3        | 3             | 6.091         | 5.579     |
| 4        | 3             | 7.842         | 7.766     |
| 5        | 3             | 10.038        | 10.223    |

### Every-action reward environment ###
| num_dice | num_dice_face | base (random) | Qlearning |
|----------|---------------|---------------|-----------|
| 3        | 2             | 4.485         | 3.965     |
| 4        | 2             | 5.901         | 6.24      |
| 5        | 2             | 7.462         | 8.109     |
| 3        | 3             | 6.091         | 5.579     |
| 4        | 3             | 7.842         | 7.766     |
| 5        | 3             | 10.038        | 10.223    |

The reason why I tried the `last-action-only` reward environment is that I wanted to see if the Q-Learning catches the high level strategy.
For example, if I got dice as [1, 2, 3, 4] when any score category is not filled, then I will choose ones (1 * 1 = 1 point).
Because I'd like to keep the chance to fill high number score cateogies with higher numbers.
In other words, if I choose to fill fours with the above dice, then I will only get 4 points (4 * 1 = 4) with the highest score category.

However, Q-learning did not catch that kind of human strategy.