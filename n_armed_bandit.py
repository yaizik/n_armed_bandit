#best run with IPython %run -i n_armed_bandit

import numpy as np

class OneArmBandit(object):
    """A single slot machine"""
    def __init__(self, mu, sigma):
        self.mu, self.sigma = mu,sigma
    def draw(self):
        return np.random.normal(self.mu, self.sigma)


class NArmBandit(object):
    """A collection of N slot machines. Slot machine i has a mean of i and sigma of N"""
    def __init__(self,N):
        self.N=N
        self.bandits=dict()

        for i in range(N):
            self.bandits[i]=OneArmBandit(i,N)

    def draw_bandit(self,i):
        return self.bandits[i].draw()


class Player(object):
    """A player, which plays the N slot machines, and need to maximize value"""
    def __init__(self,N,epsilon):
        self.game = NArmBandit(N)
        self.epsilon,self.N = epsilon,N
        self.Q = dict()
        self.num_games=dict()
        self.CumValue = 0

        for i in range(N):
            self.Q[i]=100 #optimistic init
            self.num_games[i]=0
        self.recorded_action=[]
        self.recorded_avg_value=[]

    def play(self,games=1,epsilon_decay_factor=None,epsilon_decay_step=None):
        """The play method is the single API for the Player class.
        Arguments:
        games - number of games to play
        epsilon_decay_factor, epsilon_decay_step  -if these parameters are specified, epsilon will be multiplied by epsilon_decay_factor every epsilon_decay_step games. Therefore, epsilon_decay_factor better be <1.

        After the game, you can query the recorded_action for actions taken during the game and recorded_avg_value for mean utility/value.
        """
        for game in range(games):
            if (epsilon_decay_factor != None):
                if game % epsilon_decay_step == 0:
                    self.epsilon=  self.epsilon * epsilon_decay_factor

            exploit = np.random.random() >= self.epsilon
            bandit = None
            if exploit:
                #select which i to take
                bandit = max(self.Q, key=self.Q.get)
            else:
                #randomally select between all options
                bandit = np.random.randint(0,self.N)

            r = self.game.draw_bandit(bandit)
            #update Q value:
            self.Q[bandit] = self.Q[bandit] + (1.0 / (1.0 + self.num_games[bandit])) * (r - self.Q[bandit])

            self.num_games[bandit]=self.num_games[bandit]+1
            self.CumValue+=r
            self.recorded_avg_value.append(self.CumValue / float(game+1))
            self.recorded_action.append(bandit)

NUM_GAMES = 5000
N=200

player_fixed_epsilon_02 = Player(N=N,epsilon=0.2)
player_fixed_epsilon_02.play(NUM_GAMES)
player_fixed_epsilon_01 = Player(N=N,epsilon=0.1)
player_fixed_epsilon_01.play(NUM_GAMES)
player_fixed_epsilon_00 = Player(N=N,epsilon=0.0)
player_fixed_epsilon_00.play(NUM_GAMES)
player_decay_epsilon_01 = Player(N=N,epsilon=0.1)
player_decay_epsilon_01.play(NUM_GAMES,epsilon_decay_factor=0.99,epsilon_decay_step=100)
player_decay_epsilon_04 = Player(N=N,epsilon=0.4)
player_decay_epsilon_04.play(NUM_GAMES,epsilon_decay_factor=0.9,epsilon_decay_step=100)

plt.figure()
plt.plot(player_fixed_epsilon_02.recorded_avg_value,label='fixed epsilon=0.2')
plt.plot(player_fixed_epsilon_01.recorded_avg_value,label='fixed epsilon=0.1')
plt.plot(player_fixed_epsilon_00.recorded_avg_value,label='fixed epsilon=0.0 (greedy)')
plt.plot(player_decay_epsilon_01.recorded_avg_value,label='decay epsilon (starts at 0.1)')
plt.plot(player_decay_epsilon_04.recorded_avg_value,label='decay epsilon (starts at 0.4)')
plt.title(str(N) + "-Armed Bandit")
plt.xlabel('Game #')
plt.ylabel('Mean value')
plt.legend().draggable(True)
plt.ion()
plt.show()
