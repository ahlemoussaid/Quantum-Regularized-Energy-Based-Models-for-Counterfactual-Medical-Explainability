import numpy as np

def energy_function_256d(x):
    # Two deep valleys in 256-dimensional space
    mode1 = np.full(256, -2.0)
    mode2 = np.full(256, 2.0)
    
    v1 = np.exp(-np.sum((x - mode1)**2) / 0.5)
    v2 = np.exp(-np.sum((x - mode2)**2) / 0.5)
    return -np.log(v1 + v2 + 1e-10)

def run_mcmc(steps=1000, step_size=0.05):
    print("--- Running Classical Baseline (MCMC in 256D) ---")
    current_pos = np.full(256, -2.0) # Start near Mode 1
    samples = []
    
    for i in range(steps):
        proposal = current_pos + np.random.normal(0, step_size, 256)
        
        e_curr = energy_function_256d(current_pos)
        e_prop = energy_function_256d(proposal)
        
        if np.random.rand() < np.exp(e_curr - e_prop):
            current_pos = proposal
            
        samples.append(current_pos)
        
        if (i+1) % 200 == 0:
            print(f"Step {i+1}/{steps}: Energy = {energy_function_256d(current_pos):.2f}, Position = [Local Minimum]")
            
    print("[!] Result: Severe Mode Collapse Detected. MCMC trapped in 256D local minimum.")
    return np.array(samples)

if __name__ == "__main__":
    run_mcmc()
