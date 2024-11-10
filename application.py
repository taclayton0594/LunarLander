from src.components.model_trainer import RLModelTrainer

if __name__=="__main__":
    LunarLanderMdl = RLModelTrainer()
    LunarLanderMdl.start_RL_training()
