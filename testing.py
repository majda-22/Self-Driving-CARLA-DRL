'''
  cette methode est utilisee pour tester le model 
'''
import tensorflow as tf

def testing(map, vehicle, sensors):

    tf.reset_default_graph()
    with tf.Session() as sess:
        # I have add
        saver = tf.train.import_meta_graph('./models/model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./models/'))
        #here finich
        graph = tf.get_default_graph()
        print ("helloo from testing")
        print(graph.get_operations())
        print ("helloo from testing2")
        # for op in graph.get_operations():
        #     print("we are in the loop of graph op")
        #     print(op.name)
        print ("helloo after for")
        # inputs_ = graph.get_tensor_by_name("Agent" +"/inputs:0")#we have removeinputs_ = graph.get_tensor_by_name("DQNetwork" + "/inputs:0")
        # output = graph.get_tensor_by_name("Agent" +"/output:0")# the same
        # inputs_ = graph.get_tensor_by_name("Agent/inputs:0")
        # output = graph.get_tensor_by_name("Agent/output/BiasAdd:0") 
        inputs_ = graph.get_tensor_by_name("Agent/inputs:0")
        output = graph.get_tensor_by_name("Agent/output/BiasAdd:0")

        # inputs_ = graph.get_tensor_by_name('Agent/inputs:0')  
        # output =  graph.get_tensor_by_name('Agent/actions_:0')   
        # output = graph.get_tensor_by_name('Target/actions_:0')

        episode_reward = 0
        reset_environment(map, vehicle, sensors)

        while True:
            
            state = process_image(sensors.camera_queue)
            Qs = sess.run(output, feed_dict={inputs_: state.reshape((1, *state.shape))})
            action_int = np.argmax(Qs)
            #print(Qs)
            #print(action_int)

            car_controls = map_action(action_int, action_space)
            vehicle.apply_control(car_controls)
            reward = compute_reward(vehicle, sensors)
            episode_reward += reward
            done = isDone(reward)

            if done:
                print("EPISODE ended", "TOTAL REWARD {:.4f}".format(episode_reward))
                reset_environment(map, vehicle, sensors)
                episode_reward = 0

            else:
                time.sleep(0.25)