import asyncio

from stable_baselines3 import DQN
from poke_env.player import (
    RandomPlayer,
    MaxBasePowerPlayer,
    SimpleHeuristicsPlayer,
)

from SimpleRLPlayer import SimpleRLPlayer


from poke_env import (
    PlayerConfiguration,
)

async def main():
    #Definimos las configuraciones del agente y del oponente
    opponent_1_configuration = PlayerConfiguration("RandomPlayer 2", None)
    opponent_2_configuration = PlayerConfiguration("MaxPlayer 2", None)
    opponent_3_configuration = PlayerConfiguration("SimpleHeuristic 2", None)
    rl_player_1_configuration = PlayerConfiguration("RL player 2", None)

    #Creamos a los oponentes
    opponent1 = RandomPlayer(player_configuration=opponent_1_configuration,battle_format="gen4randombattle")
    opponent2 = MaxBasePowerPlayer(player_configuration=opponent_2_configuration, battle_format="gen4randombattle")
    opponent3 = SimpleHeuristicsPlayer(player_configuration=opponent_3_configuration, battle_format="gen4randombattle")
    #Seleccionamos al rival 
    rival = opponent1

    #Creamos el entorno y el cargamos el modelo
    print('Definimos el entorno ->')
    test_env = SimpleRLPlayer(player_configuration=rl_player_1_configuration, battle_format="gen4randombattle", start_challenging=True, opponent = rival)
    print('Cargamos el modelo ->')
    model_test = DQN.load("DQN_30000i")

    #Testeamos el modelo entrenado enfrentandolo 500 veces contra el rival
    #que se ha seleccionado antes. Se guarda la recompensa promedia y el 
    #porcentaje final de victorias para la evaluación
    print('Testeamos->')
    episodios = 500 #El número de veces que queremos que se repita la simulación, en este caso serán 5 veces 
    score_final = 0
    for episodio in range(1, episodios+1):        
        observation = test_env.reset() #Restablece el entorno y devuelve la observación inicial
        terminated = False
        score = 0 #Variable para almacenar la recompensa de cada step
        
        while not terminated :
            action, state = model_test.predict(observation, deterministic=True) #El modelo predice el siguiente movimiento
            observation, reward, terminated, info = test_env.step(action) 
            score += reward
        score_final += score
    print("Resulados del modelo DQN_30000i contra el oponente: ", rival)
    print(" WIN % = ", env.n_won_battles/episodios*100)
    print(" Recompensa promedia: ", score_final/episodios)
    print('fin')
    env.close()
    

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
