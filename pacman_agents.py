import random
from collections import defaultdict

class QLearningAgent:
    """Implement Q Reinforcement Learning Agent using Q-table."""

    def __init__(self, game, discount, learning_rate, explore_prob):
        """Store any needed parameters into the agent object.
        Initialize Q-table.
        """
        self.game = game
        self.discount = discount
        self.learning_rate = learning_rate
        self.explore_prob = explore_prob
        self.q_table = defaultdict(float)  # Init Q-table as a dictionary

    def get_q_value(self, state, action):
        """Retrieve Q-value from Q-table.
        For an never seen (s,a) pair, the Q-value is by default 0.
        """
        # check if state-action pair exists, return 0 if not
        if (state, action) in self.q_table:
            return self.q_table[(state, action)]
        else:
            return 0

    def get_value(self, state):
        """Compute state value from Q-values using Bellman Equation.
        V(s) = max_a Q(s,a)
        """
        # return 0 when no actions available
        actions = self.game.get_actions(state)

        if not actions:
            return 0

        # Find max Q within all possible actions
        q_values = [self.get_q_value(state, action) for action in actions]

        return max(q_values)

    def get_best_policy(self, state):
        """Compute the best action to take in the state
        using Policy Extraction.

        π(s) = argmax_a Q(s,a)

        If there are ties, return a random one for better performance.
        Hint: use random.choice().
        """
        # get all available actions from curr state
        actions = self.game.get_actions(state)

        # return None, if no available actions
        if not actions:
            return None

        # find action with the largest Q
        best_q_value = float('-inf')
        best_actions = []

        for action in actions:
            q_value = self.get_q_value(state, action)

            # action has larger Q than curr best
            if q_value > best_q_value:
                best_q_value = q_value
                best_actions = [action]

            # action has the same Q-value as curr best
            elif q_value == best_q_value:
                best_actions.append(action)

        # return random action from the best actions (tie breaker)
        return random.choice(best_actions)

    def update(self, state, action, next_state, reward):
        """Update Q-values using running average.
        Q(s,a) = (1 - α) Q(s,a) + α (R + γ V(s'))
        Where α is the learning rate, and γ is the discount.

        Note: You should not call this function in your code.
        """
        # get the target value: R + γ V(s')
        target = reward + self.discount * self.get_value(next_state)

        # get curr Q for same state-action pair
        current_q = self.get_q_value(state, action)

        # update Q with running avg formula
        updated_q = (1 - self.learning_rate) * \
            current_q + self.learning_rate * target

        # store updated Q in the Q-table
        self.q_table[(state, action)] = updated_q

    # Epsilon Greedy
    def get_action(self, state):
        """Compute the action to take for the agent, incorporating exploration.
        That is, with probability ε, act randomly.
        Otherwise, act according to the best policy.

        Hint: use random.random() < ε to check if exploration is needed.
        """
        # get available actions from state
        actions = self.game.get_actions(state)

        # return None if no actions available
        if not actions:
            return None

        # choose a random action (exploration)
        if random.random() < self.explore_prob:
            return random.choice(list(actions))

        # choose the best action according to policy (exploitation)
        # (reuses get_policy() which selects the action with largest Q
        best_action = self.get_best_policy(state)

        if best_action is None:
            return random.choice(list(actions))

        return best_action


# Bridge Crossing
def question3():
    epsilon = 0.5
    learning_rate = 0.5
    return 'NOT POSSIBLE'


class ApproximateQAgent(QLearningAgent):
    """Implement Approximate Q Learning Agent using weights."""

    def __init__(self, *args, extractor):
        """Initialize parameters and store the feature extractor.
        Initialize weights table."""

        super().__init__(*args)

        # Feature extractor function
        self.extractor = extractor

        # Dictionary to store feature weights
        self.weights = {}

    def get_weight(self, feature):
        """Get weight of a feature.
        Never seen feature should have a weight of 0.
        """
        return self.weights.get(feature, 0)

    def get_q_value(self, state, action):
        """Compute Q value based on the dot product
        of feature componentsand weights.
        Q(s,a) = w_1 * f_1(s,a) + w_2 * f_2(s,a) + ... + w_n * f_n(s,a)
        """
        features = self.extractor(state, action)

        return sum(self.get_weight(f) * value for f, value in features.items())

    def update(self, state, action, next_state, reward):
        """Update weights using least-squares approximation.
        Δ = R + γ V(s') - Q(s,a)
        Then update weights: w_i = w_i + α * Δ * f_i(s, a)
        """
        # calculate the difference or TD error
        difference = (reward + self.discount * self.get_value(next_state)
                      - self.get_q_value(state, action))

        # feature representation for (state, action) pair
        features = self.extractor(state, action)

        # update weights via learning rule
        for feature, value in features.items():
            self.weights[feature] = self.get_weight(feature) + \
                self.learning_rate * difference * value
