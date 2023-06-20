import asyncio
import numpy as np
import time

from gym.spaces import Space, Box
from gym.utils.env_checker import check_env

from stable_baselines3 import DQN

from poke_env.environment import AbstractBattle
from poke_env.player import (
    background_evaluate_player,
    background_cross_evaluate,
    Gen4EnvSinglePlayer,
    RandomPlayer,
    MaxBasePowerPlayer,
    ObservationType,
    SimpleHeuristicsPlayer,
)

from SimpleRLPlayer import SimpleRLPlayer
from poke_env import (
    PlayerConfiguration
)


rl_player_1_configuration = PlayerConfiguration("RL player 1", None)
rl_player_2_configuration = PlayerConfiguration("RL player 2", None)
rl_player_3_configuration = PlayerConfiguration("RL player 3", None)

opponent_test_configuration = PlayerConfiguration("RandomPlayer Test", None)
opponent_test_configuration_MAX = PlayerConfiguration("MAXPlayer Test", None)
opponent_test_configuration_SH = PlayerConfiguration("SHPlayer Test", None)

opponent_test_RP = RandomPlayer(player_configuration=opponent_test_configuration,battle_format="gen4randombattle")
opponent_test_MaxBPP = MaxBasePowerPlayer(player_configuration=opponent_test_configuration_MAX,battle_format="gen4randombattle")
opponent_test_SHP = SimpleHeuristicsPlayer(player_configuration=opponent_test_configuration_SH,battle_format="gen4randombattle")

async def main():

    opponent1 = RandomPlayer(battle_format="gen4randombattle")
    opponent2 = RandomPlayer(battle_format="gen4randombattle")
    opponent3 = RandomPlayer(battle_format="gen4randombattle")
    opponent4 = MaxBasePowerPlayer(battle_format="gen4randombattle")
    opponent5 = MaxBasePowerPlayer(battle_format="gen4randombattle")
    opponent6 = SimpleHeuristicsPlayer(battle_format="gen4randombattle")
    oponentes = [opponent1,opponent4, opponent6]
    #oponentes = [opponent1,opponent2,opponent3,opponent4,opponent5, opponent6]
    #oponentes = [opponent1]

    print("Definimos entorno")  
    env = SimpleRLPlayer(battle_format="gen4randombattle", start_challenging=True, opponent = oponentes)

    print('Cargamos el modelo preentrenado ->')
    model = DQN.load("200000i")
    model.load_replay_buffer("200000i")
    
    print('Entrenamos de nuevo el modelo ->')
    inicio = time.time()
    model.set_env(env)
    model.learn(total_timesteps=300000,progress_bar=True, reset_num_timesteps = False)#, reset_num_timesteps = False
    model.save("500000i")
    model.save_replay_buffer("500000i")
    fin = time.time()
    print(' -> Modelo entrenado y guardado')
    print("Tiempo transcurrido = ", fin - inicio)
    print(f"  the model has {model.replay_buffer.size()} transitions buffer")
    del model
    
    #------------------------------------------------------------------------------------------------------------------------
    print("#---------------------------------------------------- ")
    #------------------------------------------------------------------------------------------------------------------------
    print(" Resulados del modelo 500000i")
    
    print("Testeamos el modelo contra RandomPlayer")

    model_test = DQN.load("500000i")
    test_env_RP =  SimpleRLPlayer(player_configuration=rl_player_1_configuration,battle_format="gen4randombattle", start_challenging=True,opponent=opponent_test_RP)    

    episodios = 300 #El número combates que queremos realizar
    n_battles_win = 0
    score_final = 0 #Variable para guardar la recompensa final despues de todos los episodios
    for episodio in range(1, episodios+1):        
        observation = test_env_RP.reset() #Restablece el entorno y devuelve la observación inicial
        terminated = False
        score = 0 #Variable para almacenar la recompensa de cada step
        
        while not terminated :
            action, state = model_test.predict(observation, deterministic=True) #El modelo predice el siguiente movimiento
            observation, reward, terminated, info = test_env_RP.step(action) 
            score += reward
        score_final += score
    n_battles_win = test_env_RP.n_won_battles/episodios*100
    print("  WIN % = ", n_battles_win)
    print("  Recompensa promedia: ", score_final/episodios)

    del  model_test
    
    #------------------------------------------------------------------------------------------------------------------------       
    print("#---------------------------------------------------- ")
    #------------------------------------------------------------------------------------------------------------------------
    
    print("Testeamos el modelo contra MaxBasePowerPlayer")
    model_test = DQN.load("500000i")
    test_env_MaxBPP =  SimpleRLPlayer(player_configuration=rl_player_2_configuration,battle_format="gen4randombattle", start_challenging=True,opponent=opponent_test_MaxBPP)    
    
    episodios = 300 #El número combates que queremos realizar
    n_battles_win = 0
    score_final = 0 #Variable para guardar la recompensa final despues de todos los episodios
    for episodio in range(1, episodios+1):        
        observation = test_env_MaxBPP.reset() #Restablece el entorno y devuelve la observación inicial
        terminated = False
        score = 0 #Variable para almacenar la recompensa de cada step
        
        while not terminated :
            action, state = model_test.predict(observation, deterministic=True) #El modelo predice el siguiente movimiento
            observation, reward, terminated, info = test_env_MaxBPP.step(action) 
            score += reward
        score_final += score
    n_battles_win = test_env_MaxBPP.n_won_battles/episodios*100
    print("  WIN % = ", n_battles_win)
    print("  Recompensa promedia: ", score_final/episodios)

    del  model_test
    
    #------------------------------------------------------------------------------------------------------------------------
    print("#---------------------------------------------------- ")
    #------------------------------------------------------------------------------------------------------------------------
    
    print("Testeamos el modelo contra SimpleHeuristicsPlayer")
    model_test = DQN.load("500000i")
    test_env_SHP =  SimpleRLPlayer(player_configuration=rl_player_3_configuration,battle_format="gen4randombattle", start_challenging=True,opponent=opponent_test_SHP)    
    
    episodios = 300 #El número combates que queremos realizar
    n_battles_win = 0
    score_final = 0 #Variable para guardar la recompensa final despues de todos los episodios
    for episodio in range(1, episodios+1):        
        observation = test_env_SHP.reset() #Restablece el entorno y devuelve la observación inicial
        terminated = False
        score = 0 #Variable para almacenar la recompensa de cada step
        
        while not terminated :
            action, state = model_test.predict(observation, deterministic=True) #El modelo predice el siguiente movimiento
            observation, reward, terminated, info = test_env_SHP.step(action) 
            score += reward
        score_final += score
    n_battles_win = test_env_SHP.n_won_battles/episodios*100
    print("  WIN % = ", n_battles_win)
    print("  Recompensa promedia: ", score_final/episodios)
    print('fin')

    del  model_test

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
