from managers.manager import Manager

manager = Manager(download_data=False)

manager.preprocess_data("VEA", "VWO", non_stationary_method='auto')

manager.load_ppo()

manager.train()