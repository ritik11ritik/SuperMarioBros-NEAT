import retro
import numpy as np
import cv2
import neat
import pickle
import argparse
from neat.parallel import ParallelEvaluator

game = 'SuperMarioBros-Nes'
NUM_WORKERS = 8
CHECKPOINT_GENERATION_INTERVAL = 10
CHECKPOINT_PREFIX = 'Mario'
MAX_GENS = 500
RENDER_TESTS = False
config = 'config'
test_n = 10
TEST_MULTIPLIER = 1

# Game Environment
env = retro.make(game=game, state='Level1-1', record=False)

oneD_img = []

def print_config_info():
    print("Running game: {}".format(game))
    print("Running with {} workers".format(NUM_WORKERS))
    print("Running with checkpoint prefix: {}".format(CHECKPOINT_PREFIX))
    print("Running with {} max generations".format(MAX_GENS))
    print("Running with test rendering: {}".format(RENDER_TESTS))
    print("Running with config file: {}".format(config))
    print("Running with test multiplier: {}".format(TEST_MULTIPLIER))

def parse_args():
    global CHECKPOINT_GENERATION_INTERVAL
    global CHECKPOINT_PREFIX
    global NUM_WORKERS
    global MAX_GENS
    global config
    global RENDER_TESTS

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', nargs='?', default=None,
                        help='The filename for a checkpoint file to restart from')
    parser.add_argument('--workers', nargs='?', type=int, default=NUM_WORKERS, help='How many process workers to spawn')
    parser.add_argument('--gi', nargs='?', type=int, default=CHECKPOINT_GENERATION_INTERVAL,
                        help='Maximum number of generations between save intervals')
    parser.add_argument('--checkpoint-prefix', nargs='?', default=CHECKPOINT_PREFIX,
                        help='Prefix for the filename (the end will be the generation number)')
    parser.add_argument('-g', nargs='?', type=int, default=MAX_GENS, help='Max number of generations to simulate')
    parser.add_argument('--config', nargs='?', default=config, help='Configuration filename')
    parser.add_argument('--render_tests', dest='render_tests', default=False, action='store_true')
    parser.add_argument('--test_multiplier', nargs='?', type=int, default=TEST_MULTIPLIER)
    
    command_line_args = parser.parse_args()
    CHECKPOINT_GENERATION_INTERVAL = command_line_args.gi
    CHECKPOINT_PREFIX = command_line_args.checkpoint_prefix
    NUM_WORKERS = command_line_args.workers
    config = command_line_args.config
    RENDER_TESTS = command_line_args.render_tests
    MAX_GENS = command_line_args.g
    return command_line_args
    
def eval_genomes(genomes, config):
    for i in range(1):
        frame = env.reset()         					# Current Frame
        random_action = env.action_space.sample()       # Random Action
        x, y, c = env.observation_space.shape 			# c = colour
        
        # Network
        net = neat.nn.RecurrentNetwork.create(genomes,config)
        curr_max_fitness = 0
        curr_fitness = 0
        frame_no = 0        
        counter = 0
        done = False
        
        while not done:
            if RENDER_TESTS:
                env.render()                            # Render on screen
            frame_no += 1
            
            frame = cv2.resize(frame, (56,60))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = np.reshape(frame, (56,60))
            
            oneD_img = np.ndarray.flatten(frame)

            action = net.activate(oneD_img)

            frame, reward, done, info = env.step(action)
            
            curr_fitness += reward
            if curr_fitness > curr_max_fitness:
                curr_max_fitness = curr_fitness
                counter=0
            else:
                counter+=1                # Count the frames until it is successful
                
            # Train mario for max 250 frames
            if done or counter == 250:
                done = True
                print(curr_fitness)
                
            genomes.fitness = curr_fitness
    return genomes.fitness 
            
def test_genome(net):
    reward_goal = config.fitness_threshold
    print("Testing genome with target average reward of: {}".format(reward_goal))
    
    rewards = np.zeros(test_n)
    
    for i in range(test_n * TEST_MULTIPLIER):
        print("--> Starting test episode trial {}".format(i + 1))
        frame = env.reset()
        frame = cv2.resize(frame, (56,60))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = np.reshape(frame, (56,60))
        oneD_img = np.ndarray.flatten(frame)
        
        action = net.activate(oneD_img)
        
        done = False
        t=0
        reward_episode = 0
        
        while not done:
            if RENDER_TESTS:
                env.render()
                
            frame, reward, done, info = env.step(action)
            frame = cv2.resize(frame, (56,60))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = np.reshape(frame, (56,60))
            oneD_img = np.ndarray.flatten(frame)
        
            action = net.activate(oneD_img)
            
            reward_episode += reward
            
            t+=1
            
            if done:
                print("<-- Test episode done after {} time steps with reward {}".format(t + 1, reward_episode))
                pass
            
        rewards[i % test_n] = reward_episode
        
        if i+1 >= test_n:
            average_reward = np.mean(rewards)
            print("Average reward for episode {} is {}".format(i + 1, average_reward))
            if average_reward >= reward_goal:
                print("Hit the desired average reward in {} episodes".format(i + 1))
                break
                
            
def run(checkpoint, eval_genomes):
    print_config_info()

    pe = ParallelEvaluator(NUM_WORKERS,eval_genomes)

    if checkpoint is not None:
        print("Resuming from checkpoint: {}".format(checkpoint))
        p = neat.Checkpointer.restore_checkpoint(checkpoint)
    else:
        print("Training from Scratch...")
        p = neat.Population(config)
    
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(CHECKPOINT_GENERATION_INTERVAL,
                                     filename_prefix=CHECKPOINT_PREFIX))
    
    winner = p.run(pe.evaluate, n=MAX_GENS)
    net = neat.nn.RecurrentNetwork.create(winner,config)
    
    test_genome(net)
    
    print("Finishing...")

if __name__ == '__main__':
    config = neat.Config(neat.DefaultGenome, 
                         neat.DefaultReproduction,
                         neat.DefaultSpeciesSet,
                         neat.DefaultStagnation,
                         'config')
    
    command_line_args = parse_args()
    checkpoint = command_line_args.checkpoint
    
    run(checkpoint,eval_genomes)
    env.close()