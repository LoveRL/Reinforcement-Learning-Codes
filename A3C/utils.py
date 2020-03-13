import torch
import numpy as np
import tqdm as tq
from torch import nn

'''
Transformation 'np.array' to 'torch.tensor'
'''
def To_tensor(np_array, dtype=np.float32) :
    if np_array.dtype != dtype :
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


'''
Initialize weights of Layers
'''
# Initialize method affects performance
def Init_Layers(layers) :
    for layer in layers :
        nn.init.normal_(layer.weight, mean=0.0, std=0.1)
        nn.init.constant_(layer.bias, 0.)
    return


'''
Push : Gradient of Local_A2C  ---> Gradient of Global_A2C
Pull : Parameter of Local_A2C <--- Parameter of Global_A2C
'''
def Push_and_Pull(Optimizer, Local_A2C, Global_A2C, done, next_state, state_batch, action_batch, reward_batch, discount_factor) :

    '''
    Push : Gradient of Local_A2C  ---> Gradient of Global_A2C
    '''
    # when a next_state_value is Terminal state, the value of it must be 0.
    
    # 'np.array[None, :]' : adds 1 more-dimension to the first dim index
    # 'np.array[:, None]' : adds 1 more-dimension to the second dim index
    next_state_value = 0. if done else Local_A2C.forward(To_tensor(next_state[None, :]))[-1].data.numpy()[0, 0]

    # Calculate TD-targets and append them to a buffer 
    td_target_buffer = []
    for reward in reward_batch[::-1] :
        next_state_value = reward + discount_factor * next_state_value
        td_target_buffer.append(next_state_value)
    td_target_buffer.reverse()

    # Calculate total loss of mini-batches (buffers)
    loss = Local_A2C.loss_function(
        To_tensor(np.vstack(state_batch)),
        To_tensor(np.array(action_batch), dtype=np.int64) if action_batch[0].dtype == np.int64 else To_tensor(np.vstack(action_batch)),
        To_tensor(np.array(td_target_buffer)[:, None])
        )

    Optimizer.zero_grad() # Initialize Optimizer
    loss.backward()       # Calculate gradient of loss function
    
    torch.nn.utils.clip_grad_norm_(Local_A2C.parameters(), 20) # Gradient clipping
    
    # 'x.grad' : brings x-partial gradient value of loss function
    for local_para, global_para in zip(Local_A2C.parameters(), Global_A2C.parameters()) :
        global_para._grad = local_para.grad
    # Do Optimize the loss function
    Optimizer.step()

    '''
    Pull : Parameter of Local_A2C <--- Parameter of Global_A2C
    '''
    Local_A2C.load_state_dict(Global_A2C.state_dict())

    return


'''
When the episode for each thread ends, 
Update the global variables in shared memory : Global_epi_cnt / Global_epi_reward / Result_queue
'''
def record(Global_epi_cnt, Global_epi_reward, Local_epi_reward, Result_queue, name) :

    # '.get_lock()' : to prevent multi-processes from updating
    # '.value' : to acess global variables in shared memory
    with Global_epi_cnt.get_lock() :
        
        Global_epi_cnt.value += 1
        
    with Global_epi_reward.get_lock() :
        
        if Global_epi_reward.value == 0. :
            Global_epi_reward.value = Local_epi_reward
        else:
            Global_epi_reward.value = Global_epi_reward.value * 0.99 + Local_epi_reward * 0.01

    Result_queue.put(Global_epi_reward.value)
    
    return

'''
Test the trained model
'''
def test_model(Global_Model, env) :
    
    NUM_EPI = 100
    MAX_STEPS_EPI = 200
    Result_List = []
    
    for epi in tq.tqdm(range(NUM_EPI)) :

        state = env.reset()
        done = False
        rewards_epi = 0

        for step in range(1, MAX_STEPS_EPI) :

            env.render()
            
            action = Global_Model.action_selection(To_tensor(state[None, :]))
            next_state, reward, done, _ = env.step(action.clip(-2, 2))
            
            rewards_epi += reward

            if done or step == MAX_STEPS_EPI-1 :
                Result_List.append(rewards_epi)
                break

            state = next_state

    Show_Result(Result_List, False)
    
    return


'''
Plotting Result
'''
def Show_Result(res, TRAIN=True) :
    
    import matplotlib.pyplot as plt
    import pandas as pd
    
    if TRAIN :
        plt.plot(res)
        
    elif not TRAIN :
        plt.plot(pd.DataFrame(res).rolling(10, min_periods=1).mean())
                                              
    plt.ylabel('Moving Average Epi- Reward')
    plt.xlabel('Step')
    plt.show()
    
    return
