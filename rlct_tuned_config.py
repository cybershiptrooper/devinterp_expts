from rlct_tune_utils import ChainConfig, SGLDConfig, SGNHTConfig
train_data_size = 60000
hyper_params = {
   [512, 512]: {
        "chain": ChainConfig(
            num_chains=1,
            num_draws=400,
            num_burnin_steps=0,
            num_steps_bw_draws=1,
            verbose=False,
        ),
        "optim": SGNHTConfig(
            lr=1e-5,
            diffusion_factor=0.1,
            num_samples=train_data_size,
            temperature="adaptive",
        ),
   },
}