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
# Initialize method could affects performance
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
Test the trained model
'''
def test_model(Global_Model, env) :

    print('\n >> Test Begin...')
    
    NUM_EPI = 30
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

    Show_Result(Result_List)
    
    return


'''
Plotting Result
'''
def Show_Result(res) :
    
    import matplotlib.pyplot as plt
    import pandas as pd
    import uuid
    
    file_name = 'Result - '+str(uuid.uuid4())[:8]
    plt.figure(figsize=(27, 13))                                      
    plt.ylabel('Epi- Reward', size=15)
    plt.xlabel('Step', size=15)
    plt.plot(res, marker='^')

    fig = plt.gcf()
    fig.savefig(file_name+'.png')
    plt.clf()
    plt.close()
    
    return
