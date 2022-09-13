
def get_rectangle_for_point(x, y):

    return
"""
max_num_steps = 1500
max_num_iterations = 250


for iteration in range(max_num_iterations+1):
    observation = env.reset()
    for step in range(num_steps):
        # take random action, but you can also do something more intelligent
        # action = my_intelligent_agent_fn(obs) 
        action = env.action_space.sample()
        
        # apply the action
        obs, reward, done, info = env.step(action)
        
        # Render the env
        env.render()

        # Wait a bit before the next frame unless you want to see a crazy fast video
        sleep(0.001)
        
        # If the epsiode is up, then start another one
        if done:
            env.reset()

# Close the env
env.close()

obs_space = env.observation_space #Tuples of two numbers: velocity and position
action_space = env.action_space 



print("The observation space: {}".format(obs_space))
print("The action space: {}".format(action_space))
"""