# Lessons I've Learned :pencil:

## Quantifying Sampler Performance :chart_with_upwards_trend:
- When generating reference draws with Stan's `NUTS` sampler, ensure we have sufficiently sampled from the entire distribution.
    - Increase sampling quality: set the target acceptance probability to be high (95%) and generate an immense amount of draws (1 billion)
    - Confirm symmetries are respected: ensure identical parameters have similar estimated parameter values
    - Confirm analytically-known parameters are accurately estimated: if a parameter's mean is known to be zero, the estimated mean should be near zero
    - Thin samples until effective sample size is approximately one
- Measure sampler performance with expected squared error instead of effective sample size when have access to reference draws / when not sure if converging to the proper stationary distribution
    - Cannot average expected squared error across chains b/c it is not a linear operation. Instead, concatonate every 1000 chains and compute expected squared error of this larger chain.
- Measure expected squared error of model parameters _and_ model parameters squared
    - Model parameters measure mean
    - Model parameters squared measure variance
- Use _relative_ expected squared error to compare across model parameters, unless true model parameters are close to zero because it causes division by zero
- Ensure a sampler performs well on every parameter, not just a single parameter
- Prioritize sampler performance on the slowest parameter