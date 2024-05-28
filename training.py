'''
cette classe est utilisee pour trainer le model
'''
import tensorflow as tf 
def training(map, vehicle, sensors):
    
    
    tf.reset_default_graph()
    agent = DQNetwork(state_size, action_size, learning_rate, name="Agent")
    target_agent = DQNetwork(state_size, action_size, learning_rate, name="Target")
    print("NN init")
    writer = tf.summary.FileWriter("summary")
    tf.summary.scalar("Loss", agent.loss)
    write_op = tf.summary.merge_all()
    dqn_scores = []
    eps_loss = []
    
    saver = tf.train.Saver()

    
    #init memory 
    print("memory init")

    #begin filling up memory by setting car on autopilot 
    memory = Memory(max_size = memory_size, pretrain_length = pretrain_length, action_space = action_space)
    memory.fill_memory(map, vehicle, sensors.camera_queue, sensors, autopilot=True)
    memory.save_memory(memory_save_path, memory) #save memory to sample from 
    
    with tf.Session() as sess:
        print("session beginning")
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        m = 0
        decay_step = 0
        tau = 0
        print("beginning training")
        for episode in range(1, total_episodes):
            #init episode
            print("env reset")
            #reset environment, process input state using camera sensor
            reset_environment(map, vehicle, sensors)
            state = process_image(sensors.camera_queue)
            done = False
            start = time.time()
            episode_reward = 0
        
            #step through episode & retrieve data from DNN
            for step in range(max_steps):
                #increment tau and decay to account for updating target 
                #features and improving epsilon-greedy policy
                
                #Logic is that as agent trains more in an episode
                #it will learn more optimal actions and thus does not need
                #to take random actions as often
                #Require lower explore_prob to drive agent towards choosing
                #greedy actions more often as it is trained more 
                
                tau += 1
                decay_step += 1
                #return optimal action index (action_int), action's one-hot encoding (action), and explore_prob
                action_int, action, explore_probability = agent.predict_action(sess, explore_start, explore_stop, decay_rate, decay_step, state)
                print("action from NN received")
                #pass in action index & all possible actions from space, map it to carla control
                car_controls = map_action(action_int, action_space)
                vehicle.apply_control(car_controls)
                print("action applied to car")
                time.sleep(0.25)
                next_state = process_image(sensors.camera_queue)
                reward = compute_reward(vehicle, sensors)
                print("reward computed: " + str(reward))
                
                #compute reward, add experience to memory and inc to next state
                episode_reward += reward
                done = isDone(reward)
                memory.add((state, action, reward, next_state, done))
                state = next_state
            
                #begin learning by sampling a batch from memory
                #remember every experience is an array
                #when sampling a batch, useful to split up experiences into
                #constituent arrays to access individual samples for q-learning
                batch = memory.sample(batch_size)
                
                #state samples for batch
                s_mb = np.array([each[0] for each in batch], ndmin = 3)
                #action samples for batch
                a_mb = np.array([each[1] for each in batch])
                #reward samples for batch
                r_mb = np.array([each[2] for each in batch])
                #next state samples for batch
                next_s_mb = np.array([each[3] for each in batch], ndmin = 3)
                #done flag samples for batch
                dones_mb = np.array([each[4] for each in batch])
                
                target_Qs_batch = []

                #q-val for all next states to compute target q-val for current state
                Qs_next_state = sess.run(agent.output, feed_dict={agent.inputs_: next_s_mb})
                Qs_target_next_state = sess.run(target_agent.output, feed_dict={target_agent.inputs_: next_s_mb})
                
                for i in range(0, len(batch)):
                    terminal = dones_mb[i] #check if on last state of eps
                    action = np.argmax(Qs_next_state[i]) #store index of optimal action
                    if terminal:
                        target_Qs_batch.append((r_mb[i])) #if last state, append reward
                    else:
                        #formulate target q-vals by feed-fwd in network, using old weights for comparison 
                        #choose optimal action & compute q-val via target net rather than use argmax & same net here
                        #reduces overestimation
                        target = r_mb[i] + gamma*Qs_target_next_state[i][action]
                        #target = r_mb[i] + gamma*np.max(Qs_next_state[i])
                        target_Qs_batch.append(target)
                targets_mb = np.array([each for each in target_Qs_batch])
                
                #run session to compute loss & change NN weights, "newly" learned weights compared against old ones in training
                #feed in state inputs, actions to associate q-vals w/, and target q-vals for loss computation
                loss, _  = sess.run([agent.loss, agent.optimizer], feed_dict={agent.inputs_: s_mb, agent.target_Q: targets_mb, agent.actions_:a_mb})
                summary = sess.run(write_op, feed_dict={agent.inputs_: s_mb, agent.target_Q: targets_mb, agent.actions_:a_mb})
                writer.add_summary(summary, episode)
                writer.flush
                
                if tau > max_tau: #update target net weights every 5000 steps/actions 
                    update_target = update_target_graph()
                    sess.run(update_target)
                    m += 1
                    tau = 0
                    print("model updated")
                
                if episode % 5 == 0:
                    save_path = saver.save(sess, "./models/model.ckpt")
                    save_path
                    print("Model Saved")
                
                print('Episode: {}'.format(episode),
                                  'Total reward: {}'.format(episode_reward),
                                  'Explore P: {:.4f}'.format(explore_probability),
                                'Training Loss {:.4f}'.format(loss))
                
                if sensors.collision_flag == True: #if vehicle collides, reset
                    break
                
            #track loss and rewards to analyze data 
            eps_loss.append(loss)
            dqn_scores.append(episode_reward)
        print("Loss per episode")
        print(eps_loss)
        print("Reward per episode")
        print(dqn_scores)