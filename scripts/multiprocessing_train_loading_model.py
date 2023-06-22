import asyncio

import gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import DQN

from poke_env.player import (
    RandomPlayer,
    MaxBasePowerPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env import (
    PlayerConfiguration,
)
from SimpleRLPlayer import SimpleRLPlayer

import time

#Definimos las configuraciones del agente y del oponente
opponent_1_configuration = PlayerConfiguration("RandomPlayer 1", None)
opponent_2_configuration = PlayerConfiguration("RandomPlayer 2", None)
opponent_3_configuration = PlayerConfiguration("RandomPlayer 3", None)
opponent_4_configuration = PlayerConfiguration("RandomPlayer 4", None)
opponent_5_configuration = PlayerConfiguration("MaxPlayer 1", None)
opponent_6_configuration = PlayerConfiguration("MaxPlayer 2", None)
opponent_7_configuration = PlayerConfiguration("MaxPlayer 3", None)
rl_player_1_configuration = PlayerConfiguration("RL player 1", None)
rl_player_2_configuration = PlayerConfiguration("RL player 2", None)
opponent_test_configuration = PlayerConfiguration("MaxPlayer Test", None)
    
#Creamos a los oponentes
opponent1 = RandomPlayer(player_configuration=opponent_1_configuration,battle_format="gen4randombattle")
opponent2 = RandomPlayer(player_configuration=opponent_2_configuration,battle_format="gen4randombattle")
opponent3 = RandomPlayer(player_configuration=opponent_3_configuration,battle_format="gen4randombattle")
opponent4 = RandomPlayer(player_configuration=opponent_4_configuration,battle_format="gen4randombattle")
opponent5 = MaxBasePowerPlayer(player_configuration=opponent_5_configuration,battle_format="gen4randombattle")
opponent6 = MaxBasePowerPlayer(player_configuration=opponent_6_configuration,battle_format="gen4randombattle")
opponent7 = MaxBasePowerPlayer(player_configuration=opponent_7_configuration,battle_format="gen4randombattle")
opponent_test = MaxBasePowerPlayer(player_configuration=opponent_test_configuration,battle_format="gen4randombattle")

async def main():
    #Definimos una variable para almacenar los oponentes del agente
    rival = [opponent1, opponent5]
    
    #Definimos los entornos de entrenamiento y de test
    print("Definimos entorno") 
    train_env = SimpleRLPlayer(player_configuration=rl_player_1_configuration,battle_format="gen4randombattle", start_challenging=True,opponent=rival)    
    test_env =  SimpleRLPlayer(player_configuration=rl_player_2_configuration,battle_format="gen4randombattle", start_challenging=True,opponent=opponent_test)    

    #Vectorizamos el entorno
    print("Vectorizamos entorno") 
    num_envs = 50  #Numero de entornos a usar
    #Por defecto, se usa DummyVecEnv
    vec_env = make_vec_env(lambda: train_env, n_envs=num_envs)

    print("Reentrenamos el modelo")
    inicio = time.time()
    #Parametros a modificar
    #   policy: La politica del modelo a utilizar
    #   env: El entorno del que aprender
    #   verbose: - 0 for no output 
    #            - 1 for info messages
    #            - 2 for debug messages
    #   learning_starts: cuanto pasos está recopilando informacion el modelo
    #                    antes de empezar a entrenar
    #   target_update_interval: cada cuantos steps se actualiza la red neuronal objetivo
    initial_training_steps = 20001
    new_training_steps = 10001
    steps_model = 20001
    model = DQN.load("DQN_"+str(initial_training_steps)+"i", learning_starts=10000, target_update_interval=10000)
    model.load_replay_buffer("DQN_"+str(initial_training_steps)+"i")
    model.set_env(vec_env)
    model.learn(total_timesteps=new_training_steps,progress_bar=True, reset_num_timesteps = False)
    steps_model = initial_training_steps + new_training_steps
    model.save("DQN_"+str(steps_model)+"i")
    model.save_replay_buffer("DQN_"+str(steps_model)+"i")
    fin = time.time()
    print(' -> Modelo entrenado y guardado')
    print("Tiempo transcurrido = ", fin - inicio)

    del model  

    #------------------------------------------------------------------------------------------------------------------------
    print("#---------------------------------------------------- ")
    #------------------------------------------------------------------------------------------------------------------------

    print("Testeamos el modelo")
    print(' Cargamos el modelo ->')
    model_test = DQN.load("DQN_"+str(steps_model)+"i")
    print(' Runeamos->')
    episodios = 200 #El número combates que queremos realizar
    n_battles_win = 0
    score_final = 0 #Variable para guardar la recompensa final despues de todos los episodios
    for episodio in range(1, episodios+1):        
        observation = test_env.reset() #Restablece el entorno y devuelve la observación inicial
        terminated = False
        score = 0 #Variable para almacenar la recompensa de cada step

        while not terminated :
            action, state = model_test.predict(observation, deterministic=True) #dejamos al modelo precedir el siguiente movimiento
            observation, reward, terminated, info = test_env.step(action) #reward son las recompensas (+1),done=False y es True cuando es el momento de restablecer el entorno o el objetivo alcanzado, info es un diccionario para la depuración
            score += reward
        score_final += score
    print(" Resulados del modelo DQN_"+str(i)+"i contra el oponente: ", rival)
    n_battles_win = test_env.n_won_battles/episodios*100
    print("  WIN % = ", n_battles_win)
    print("  Recompensa promedia: ", score_final/episodios)
    print('fin')

    del  model_test

    #Repetimos el proceso mientras que el porcentaje de victorias sea menor al 75%
    new_training_steps = 10001
    while(n_battles_win < 75):
        print("Volvemos a entrenar al modelo DQN_"+str(steps_model)+"i")
        model = DQN.load("DQN_"+str(steps_model)+"i",target_update_interval=10000)
        model.load_replay_buffer("DQN_"+str(steps_model)+"i")
        model.set_env(vec_env)
        model.learn(total_timesteps=new_training_steps,progress_bar=True, reset_num_timesteps = False)
        steps_model += new_training_steps
        model.save("DQN_"+str(steps_model)+"i")
        model.save_replay_buffer("DQN_"+str(steps_model)+"i")
        fin = time.time()
        print(' -> Modelo entrenado y guardado')
        print("Tiempo transcurrido = ", fin - inicio)

        del model

        #------------------------------------------------------------------------------------------------------------------------
        print("------------------------------------------------------------------")
        #------------------------------------------------------------------------------------------------------------------------

        print("Testeamos el modelo")
        print(' Cargamos el modelo ->')
        model_test = DQN.load("DQN_"+str(steps_model)+"i")
        print(' Runeamos->')
        test_env.close()
        test_env.reset_battles()
        test_env.start_challenging()
        score_final = 0
        for episodio in range(1, episodios+1):        
            observation = test_env.reset() #Restablece el entorno y devuelve la observación inicial
            terminated = False
            score = 0

            while not terminated :
                action, _state = model_test.predict(observation, deterministic=True) #dejamos al modelo precedir el siguiente movimiento
                observation, reward, terminated, info = test_env.step(action) 
                score += reward
            score_final += score
        print(" Resulados del modelo DQN_"+str(steps_model)+"i contra el oponente: ", opponent_test)
        n_battles_win = test_env.n_won_battles/episodios*100
        print("  WIN % = ", n_battles_win)
        print("  Recompensa promedia: ", score_final/episodios)
        print('fin')
        
        del model_test

    

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())

