# RL Summary

### TODO 

##### Check all pseudo algorithms in sutton and barto

Questions:

1. Exploration-evaluation comparison graphs in lecture 7 are not clear
2. What to remember of GAE?
3. 



Overview:

## MDP & DP: Lecture 1-2

### Lecture 1: Markovian Decision Processes (MDP)

Key takeaways / points to remember: Which of these are Markov?

- Markov means: Next state depends only on the current state and action
- Rewards depends only on (some of) the state, action, and next state
- Discrete time steps

1. Chess (with / without opponent’s fixed reactive policy) 

   1. With opponent's reactive policy fixed

      Depends on how the state is saved. If it's saved as a sequence of transitions, then yes, as you can always reconstruct the current board, and know whether or not the opponent castled. The reward can be seen as stealing a chess piece from the opponent.

   2. Without opponent's reactive policy fixed

      Same, I think, as nothing changes in the Markov points. Intuitively I was thinking, though, that it wasn't, as you couldn't know the exact value of the next state since you don't know how the opponent responds. But is this 'the reward' or 'the expected future reward'? I think that here, also, the reward only depends on state, action, and next state, and nothing else. Calculating the expected reward of that state without a reactive policy, though, becomes more difficult.

2. Robot in maze (with / without exact position sensing) 

   1. With exact position sensing (I assume this to be GPS)

      With "exact position" sensing I assume 1. the position in the maze and 2. the direction that the robot is facing. If we assume a robot in real life, though, there could always be mechanical issues that are not sensed, and that could prevent the robot from moving in a certain direction. However, if we do not take this into account (i.e. a virtual robot in virtual maze) this would be a MDP.

   2. Without exact position sensing

      Let's assume the robot only has eye-like sensors so it doesn't know at any point in time where in the maze it is. It is then required to remember all the previous steps in order to know where in the maze it is and to make a fair decision. For example, if he sees a wall in front of him and he wants to walk to the right, at certain points in time he might bump into a wall, while at other times he would get a reward because he walked out of the maze. So the reward depends on more than just his current state ("wall in front of him") and his current action ("walk to the right").

3. Autonomous driving

   We can "simply" say that it doesn't fit the Markov Decision Process definition, as there aren't discrete time steps.

4. Ad serving

   Google is trying to make this a markov decision process by learning more and more about their users by collecting a wealth of data, but it is likely to never be fully markovian.

   We can say that the current state is "user searches for query 'x'". The algorithm then takes an action to show advertisement y. Possible next states are "user clicks ad" or "user searches for new query". Clearly, the user's state of mind and opinion on search results influence their behaviour that influence the next state, and therefore the first markov property doesn't hold. 

## Value-based, model-free RL: Lecture 2-3-4-5-6. 

1. Learn value function: V(s) or Q(s,a)
2. Optimize policy $$\pi(a|s)$$

### Lecture 2: Dynamic Programming and Monte Carlo

Q: In what situations do we not know transition distribution,  but are able to obtain experience trajectories

A: ??

Q: From some state s, for which actions will we learn a meaningful q function?

A ??

### Lecture 3: Model-free methods and importance weights

We want to move away from exploring starts

1. On-policy: Change policy update to move towards greedy policy, but keep exploring

   Start with $\epsilon$-soft, new $\epsilon$-greedy policy will be better, so GPI will converge to <u>optimal $\epsilon$-soft policy</u>

2. Off-policy: Can we do better than optimal $\epsilon$-soft policy? Use greedy target and non-greedy behaviour policy (required importance sampling).

   Assumes the coverage assumption: $\pi(a|s)>0 \rightarrow b(a|s) > 0$.

   **Importance Sampling**

   $\rho_{t:T-1} = \prod^{T-1}_{k=t}\frac{\pi(A_k|S_k)}{b(A_k|S_k)}$

   - Ordinary importance sampling: $\tilde{V}_{\pi}(s) \dot{=} \frac{\sum_{t \in T^*}\rho_{t:T(t)-1}G_t}{|T^*(s)|}$

   - Weighted importance sampling: $\tilde{V}_{\pi}(s) \dot{=} \frac{\sum_{t \in T^*}\rho_{t:T(t)-1}G_t}{\sum_{t \in T^*}\rho_{t:T(t)-1}}$

     |          | one trajectory                       | $\rho=10$ or $\rho=0.1$                 |
     | -------- | ------------------------------------ | --------------------------------------- |
     | Ordinary | good on average                      | estimates very high/low: high variance. |
     | Weighted | $\frac{\rho}{\rho}=1$, always biased | Estimates close together, low variance  |

   - First-visit ordinary IS is unbiased

   - Every-visit or weighted IS is biased

     - but much lower variance
     - typically preferred
     - easier to implement
     - bias $\rightarrow$ 0 as N increases

   

   TD-error: Instead of a full MC rollout, we look at estimated V of the next state and take the difference

   SARSA: On-policy TD control: Plug $Q(s,a)$ instead of $V(s)$

   Expected SARSA: Off-policy TD Control: Target calculated as $\sum_a \pi(a|s_{t+1})Q(s_{t+1},a) = V(s_{t+1})$ right?

### Lecture 4: Advanced TD Methods

Q: Explain why Q-learning, Expected SARSA, and n-step tree backup algorithms all do not need importance sampling while they are off-policy.

A: ?? Long thread on Slack that still didn't make it fully clear to me.

Q-learning: off-policy TD Control. Like expected sarsa, but with greedy policy instead of average over following (state, action) pairs.

**What are advantages of TD(0), N-step, and Monte-Carlo**

- TD(0), N-step TD, and Monte-Carlo all fall within the same "two classes" of algorithms: 

  1. they could be narrow prediction methods. 

     ​	\+ No need for transition dynamics (they are model-free)

  2. They could be on-policy control methods

     ​	\+ Simpler than off-policy methods

     ​	\+ Convergest faster than off-policy methods

     ​	\- Only a specific case (only for target polict $\pi$)

     ​	\- Only data from the current target policy $\pi$

     ​	\- Requires a non-greedy policy

- Between TD(0), N-step, and MC, there are also advantages:

  - TD(0)

    ​	\+ Can exploit learned values at intermediate states

  - MC

    ​	\+ can quickly back-up fro ma single episode

  - N-step TD:

    ​	\+ Best of both worlds? Might work better for different problems

**What are some off-policy and on-policy methods for TD-learning and their properties?**

- Q-learning - off-policy, updates with greedy policy in next state
- TD(0) - Uses the approximated value of the next state as a target
- SARSA - on-policy, updates with actual step taken in next state
- Expected SARSA - off-policy, updates with mean of next state
- N-step TD - In between TD(0) and MC, requires importance weights if used with a behavior policy $b$
- N-step Tree backup diagram

**What is maximisation bias?**

​	Over-estimates because of the target being $R + \gamma Q(s', {\arg \max}_{a'} Q(s',a'))$, and it taking the max.

​	Solution: double Q learning

**What is optimization bias?**

​	I believe this was mentioned during the lecture. Something about "introducing errors, and optimizing on the errors. if error comes from normal distribution, the action that happens to get a large error will be chosen as the best". Very closely related to the maximization bias.

**How can value-function methods be categorized?**

​	We can categorize value functions (prediction) and q functions (control)

- Value functions are categorized by width and depth

  |                        | Width $\rightarrow$ |      |                    |
  | ---------------------- | ------------------- | ---- | ------------------ |
  | **Depth** $\downarrow$ | TD(0)               |      | DP                 |
  |                        | N-step TD           |      |                    |
  |                        | MC                  |      | Exchaustive Search |

  

- Q functions are categorized by whether they are on- or off-policy, and how deep they are

  | On-policy           | **SARSA**                      |               | N-step TD          | MC Control                                   |
  | ------------------- | ------------------------------ | ------------- | ------------------ | -------------------------------------------- |
  | Depth $\rightarrow$ | $\rightarrow$                  | $\rightarrow$ | $\rightarrow$      | $\rightarrow$ Depth                          |
  | Off-policy          | Q-Learning<br />Expected Sarsa |               | N-step Tree backup | $\}$ all do not require importance sampling! |
  |                     |                                |               |                    |                                              |

### Lecture 5: Prediction with approximation

Gradient MC w/ approximation vs Semi-gradient TD w/ approxiation

Q: Although semi-gradient TD has much lower variance and trains faster. Why can the final results become worse when we train indefinitely?

A: ?

Linear Function approximation: $\hat v (s,\textbf w) \dot = \textbf w^T \textbf x(s) \dot = \sum_{i=1}^d w_i x_i (s)$ Convince yourself:

- What does x look like for the tilings example?
  - e.g. a one-hot vector, denoting where the continuous value lies.

- Is tabular RL a specific case of a linear function? What would x be?
  - ?? $\textbf x$ would be a one-hot encoded feature vector that denotes in which state the agent is. Then the weight connected to each entry maps it to an action to be taken in that exact state.

Q: What can we say about the divergence properties of Gradient MC and Semi-gradient TD when we use off-policy data?

### Lecture 6: Off-policy & Control with approximation

TODO: {[check simple examples from the book], [check 3d drawings of all the errors]}

Conclusions:

On-policy control straightforward

Off-policy prediction and control very tricky!

Deadly triad:

1. Function approximation					Required to scale up to large problems
2. Semi-gradient bootstrapping.           Because MC is slow
3. Off-policy training                              Because we want to. reuse old data / recorded data / learn different policies 

Alternative to semi-gradients?

1. Min TDE                                              Fails with A-split problem
2. MC Solution (Min VE)                         We don't want this, MC is not cool
3. TD Fixpoint (PBE = Mean squared projected bellman error).        This gives GTD2 (gradient TD method)
4. Smallest avg BE                                 Works for a-split problem, but can't get gradient from data

*--- Beautiful 3D representation of all the searches below... what should we know? ---*

Using gradient of PBE is the winner! But: Keep track of more values

<img src="/Users/yoni/Library/Application Support/typora-user-images/image-20191021094813249.png" alt="image-20191021094813249" style="zoom: 25%; align:left;" />	

- Gradient TD, that uses gradient of PBE, is a solution • Overhead: keep track of more values

Alternative: use additional mechanisms to keep learning stable. Good empirical performance with DQN

DQN: Makes Reinforcement Learning closer to Supervised Learning

- Experience replay
  - Break correlation between training data by saving many experiences in memory and using a random batch from memory $\rightarrow$ **more i.i.d.**
- Target network
  - Keep the target fixed by copying parameters $\textbf w $ every $C$ steps. Breaks the instability due to the depende of the target on $\textbf w$. $\rightarrow$ **fixed target** 

## Policy-based, model-free RL: Lecture 7-8-9-10

\#TODO HW2 is very unclear to me. All seem way too mathematical, and don't necessarily provide me any insights.

1. Optimize policy $$\pi(a|s)$$

- - 



### Lecture 7:  Policy-based methods: Actor-only. (TEMPORAL DIFFERENCE/REINFORCE/G(PO)MDP/PGPE)

What Policy gradients fix the problems of action-value methods?

- handle continuous actions
  - Can use policy with continuous output (linear, neural net)

- ensure smoothness in policies 
  - Small step size ensures small change in policy

- small errors in V not the same as small errors in policy
  -  Directly optimise quantity of interest

- hard to include prior knowledge about possible solutions 
  - Include prior knowledge as policy form or initialisation

- can’t learn stochastic policies 
  - Can easily train stochastic policies

Still, weaknesses:

Actor-only methods have high variance from Monte-Carlo
A lot of the methods we discussed are specific to episodic setting Requires stochastic policies, what if deterministic is optimal?

- If amount of randomness is learned, can get close to deterministic

- We will also see a policy gradient method to learn deterministic policies



### Lecture 8 Policy-based methods 2: Policy Gradient Theorem & Actor-critic methods. Staying close to previous policy: Natural Policy Gradient

Again::: **Policy** **search** methods typically preferred in any of the below cases (all there in robotics!)

1. Problems with continuous actions
2. If we need to learn stochastic policies
3. If we have prior knowledge about the type of policy
4. If it is important to have ‘small’ policy updates between subsequent time step

**Actor-critic** methods add value estimation. Instead of doing a full rollout and calculating $G_t$ ($\sum_t r_t$) from the current state (a MC method, which takes long, introduces noise for increasing $T$, and is only defined for episodic settings), they use the estimate $\hat{q}_w(s,a)$ together with $\pi_\theta(s)$.x

And **Actor-Critic** addresses some of these problems:

​	\+ Actor-only methods have high variance from Monte-Carlo
​		Actor-critic lowers variance using critic
​	\+ A lot of the methods we discussed are specific to episodic setting
​		Actor-critic can be formulated for continuing setting
​	\- Actor-critic can be ‘fiddly’, many moving parts
​	\- Requires stochastic policies, what if deterministic is optimal?
​		If amount of randomness is learned, can get close to deterministic
​		We will also see a policy gradient method to learn deterministic policies

Focus on **Natural Policy Gradient**

We use KL divergence as a measure to ensure we don't make too big steps. We get a full second order gradient, which allows us to do much smarter updates. 

### Lecture 9: Policy-based methods 3: Trust-region Policy Optimization. Generalized Advantage Estimation

TRPO Conclusion:

- Better metric for policy updates: use structure of parameters 
- Allows taking larger steps in policy space than e.g. PGT 
- NPG, NAC: easy to implement
- TRPO: larger steps (faster), use with neural network

What should you remember?

1. Advantage of covariant representation of distances? 
   - No need to tune param representation
   - Works for any representation
2. Advantage of specifying constraint instead of stepsize?
   - Can use this to model the correlation between the parameters with e.g. KL divergence $\rightarrow$ Fisher Matrix
3. Why do we need a constraint / penalty / stepsize? 
   - Because we do not want to make too big changes to the policy
     - It breaks robots
     - Since estimates can be noisy, we want to move slowly between policy updates

Generalized Advantage Estimation

**CONCLUSIONS ABOUT GAE** 

- GAE can interpolate between 
  - MC (∞-step return) used by REINFORCE 
  - TD (1-step return) used by PGT-AC
- n-step methods are comparable, but GAE averages over many different n 
- Can be tuned to problem at hand using λ and  
- Can yield huge improvement compared to either TD or MC Can be used with many policy-based algorithms 

### Lecture 10: (Deep) Deterministic Policy Gradient & Experimenting, reporting with deep methods

Remember: (see notes)

- What is the main concept behind DPG?
- What are some challenges when tackling reinforcement learning problems using deep neural networks?
- What are some main challenges in evaluating deep reinforcement learning methods?
- How to recognise and avoid problems in evaluating deep reinforcement learning methods?

Recommendations:

- Run multiple trials with different random seeds, starting from scratch each time.
- Report mean/median as well as a measure of ‘spread’
- Sanity check result of your hyperparameters & implementation with published results
- Spend roughly equal amount of effort tuning different methods Report all details required for reproduction and interpretation!
  - All hyperparameters, architecture, ‘tricks’
  - What was measured, how, #of independent runs

## Model-based RL: Lecture 11-12-13

1. Learn model $$p(s'|s,a) r(s,a)$$
2. Learn value or policy $$V(s), Q(s,a), \pi(a|s)$$
3. Optimize policy $$\pi(a|s)$$

### Lecture 11: Model-based RL

Remember:

- Why do model-based reinforcement learning?

  1. Generate proxy data from model so that we need less data from the system. Good if collecting data is expensive. E.g. when gathering data with a real robot, there has to be a supervisor. Experiments involving humans require recruitment, participation, watching, or the data is simply not available because it comes in over time as the data is collected from interaction from real users.

  2. Can give something that real data doesn't have, e.g. probability distribution of actions
  
  
      If both of the above are not the case, don't use model-based RL
  
- Model based value learning vs model based policy search

  - Any of the below can use more information from the model, e.g. gradient information.
  - Policy:
    - Policy-based search good if action space is continuous
    - Slightly different loop, but very similar to dyna-Q
      - Internal simulation $\rightarrow$ Policy Learning $\rightarrow$ Apply policy to robot $\rightarrow$ model learning $\rightarrow$ ....
    - To calculate gradient requires chain-ruling (from reward > policy > model), e.g. $\frac{\partial r}{\partial s'}\frac{\partial s' }{\partial a}\frac{\partial a}{\partial \theta}$
  - 

- What is the general structure of model-based learning

  - Acting $\rightarrow$ experience $\rightarrow$ model $\rightarrow$ planning (RL on simulated data) $\rightarrow$ value/policy $\rightarrow$ Acting ....
    - Means we can learn with fewer examples. But, we could be learning the **wrong model**.
  - Instead, value-based learning = Directl RL = model-free RL all go
    Acting $\rightarrow$ experience $\rightarrow$ value/policy $\rightarrow$ Acting ....

- What are some answers to the questions:

  - How to learn model
    - dyna-q, storing transition in table, using maximum likelihood
    - model with NN
    - 
  - When to update
    - When to Plan? E.g. when it's your turn in a game of chess/go. As you can't do this all beforehand, you take time during the move to sample possible trajectories from the current state and save those in memory, and only keep part of the memory that is relevant after the next actions have been decided.
  - What to update
    - Which states? Random states? 
    - Prioritized sweeping: Following trajectory and only update those that were updated (and the states that lead to the updated states?)
  - How to update
    - Q-learning / DP? Depends on branching factor. High branching factor, sampling with Q-learning is better. WIth low branching factor, DP will be quicker. But, DP methods require a full distribution model instead of a (simpler) sampling model. 

### Lecture 12

### Lecture 13

### Lecture 14

4. 



### 

### 

### 

### 