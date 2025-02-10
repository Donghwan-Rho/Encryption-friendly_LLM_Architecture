import matplotlib.pyplot as plt
import logging

log = logging.getLogger(__name__)

def graph_and_ratio(task):
    # Read the data from the files
    with open(f'{task}/custom_adamw_var_sqrt_input_max.txt', 'r') as f:
        max_values = [float(line.strip()) for line in f.readlines()]
    
    with open(f'{task}/custom_adamw_var_sqrt_input_min.txt', 'r') as f:
        min_values = [float(line.strip()) for line in f.readlines()]
    
    # Calculate max(max/min)
    max_max_value = max(max_values)
    min_min_value = min(min_values)
    result = max_max_value / min_min_value
    
    # # Write the result to result.txt
    # with open(f'{task}/result.txt', 'w') as f:
    #     f.write(f"max(max/min): {result:.3f}\n")
    
    # # Plot the max values and save the plot
    # plt.figure(figsize=(10, 6))
    # plt.plot(max_values, marker='o', linestyle='-')
    # plt.title('Max Values Over Step')
    # plt.xlabel('Step')
    # plt.ylabel('Max Value')
    # plt.grid(True)
    # plt.savefig(f'{task}/sqrt_input_max.png')
    # plt.close()
    
    # Write the result to result.txt
    with open(f'result.txt', 'a') as f:
        f.write(f"{task} max(max/min): {result:.3f}\n\n")
        log.info(f"{task} max(max/min): {result:.3f}")
    
    # Plot the max values and save the plot
    plt.figure(figsize=(10, 6))
    plt.plot(max_values, marker='o', linestyle='-')
    plt.title('Max Values Over Step')
    plt.xlabel('Step')
    plt.ylabel('Max Value')
    plt.grid(True)
    plt.savefig(f'{task}_sqrt_input_max.png')
    plt.close()