def compute_pure_diffusion_time(
    diffusion_time: float,
    agent_time: float = 0.0,
    memory_time: float = 0.0,
    recache_time: float = 0.0,
) -> tuple[float, float]:
    pure_time = diffusion_time - agent_time - memory_time - recache_time
    pure_pct = 0.0 if diffusion_time == 0 else 100 * pure_time / diffusion_time
    return pure_time, pure_pct
