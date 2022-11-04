from aalpy.utils import load_automaton_from_file, statistical_model_checking

dqn_model = load_automaton_from_file('mdp_dqn.dot', automaton_type='mdp')
a2c_model = load_automaton_from_file('mdp_a2c.dot', automaton_type='mdp')

dqn_model.make_input_complete()
a2c_model.make_input_complete()

print('Performing SMC')
dqn_smc = statistical_model_checking(dqn_model, {'DONE'}, max_num_steps=50)
a2c_smc = statistical_model_checking(a2c_model, {'DONE'}, max_num_steps=50)

print(f'DQN: {dqn_smc}\nA2C: {a2c_smc}')
