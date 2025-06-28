import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Choose the appropriate backend as needed
matplotlib.rcParams['font.family'] = 'Arial'  # Set font to Arial
matplotlib.rcParams['axes.unicode_minus'] = False  # Display minus sign correctly
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint
import csv

# Parameter Settings
num_niches = 100          # Number of niches
num_simulations = 1000    # Number of simulations per occupancy proportion

# Standard Deviation Settings for Environmental Changes
sigma_env_small = 1  # Small environmental changes, simulate background extinction
sigma_env_large = 5  # Large environmental changes, simulate mass extinction

# Occupancy Proportions for Category A (from 100% to 1%)
a_proportions = np.arange(100, 0, -1)  # From 100% to 1%, decreasing by 1% each step

# Dictionary to Store Results
results = {'Background Extinction': {'prob': [], 'lower': [], 'upper': []},
           'Mass Extinction': {'prob': [], 'lower': [], 'upper': []}}

# Initialize List to Store CSV Data
csv_data = []

# Generate Random Spatial Positions for Each Niche
np.random.seed(42)  # Set random seed for reproducibility
niche_positions = np.random.rand(num_niches, 2)  # Random coordinates in 2D space [0,1]x[0,1]

# Loop Through the Two Environmental Change Scenarios
for scenario, sigma_env in zip(['Background Extinction', 'Mass Extinction'], [sigma_env_small, sigma_env_large]):
    extinction_probs = []    # Store probabilities of complete extinction for Category A
    lower_bounds = []        # Store lower bounds of 95% confidence intervals
    upper_bounds = []        # Store upper bounds of 95% confidence intervals

    # Iterate Over Each Occupancy Proportion for Category A
    for a_prop in a_proportions:
        # Calculate Number of Species in Categories A and B
        num_a_species = int(num_niches * (a_prop / 100))
        num_b_species = num_niches - num_a_species

        # Randomly Assign Species to Niches (Simulate Random Spatial Distribution)
        niche_indices = np.arange(num_niches)
        np.random.shuffle(niche_indices)
        a_niches = niche_indices[:num_a_species]    # Indices of niches occupied by Category A species
        b_niches = niche_indices[num_a_species:]    # Indices of niches occupied by Category B species

        # Set Random Niche Breadth for Category A Species, assuming it ranges between min and max values
        min_niche_breadth = 0.1
        max_possible_niche_breadth = 1  # Maximum possible niche breadth

        if num_a_species > 0:
            # Assign random niche breadths to Category A species, decreasing as occupancy proportion decreases
            max_niche_breadth_a = min_niche_breadth + (max_possible_niche_breadth - min_niche_breadth) * (a_prop / 100)
            niche_breadth_a = np.random.uniform(low=min_niche_breadth, high=max_niche_breadth_a, size=num_a_species)
        else:
            niche_breadth_a = np.array([])  # Empty array when there are no Category A species

        # Set Niche Breadth for Category B to a fixed value
        niche_breadth_b = 1.0

        # Counter to Record the Number of Times Category A is Completely Extinct
        a_extinct_count = 0

        # Perform Simulations
        for sim in range(num_simulations):
            # Simulate Environmental Changes for Each Niche
            environmental_changes = np.random.normal(0, sigma_env, size=num_niches)

            # Determine Survival of Category A Species
            if num_a_species > 0:
                # Get environmental changes for niches occupied by Category A species
                env_change_a = environmental_changes[a_niches]
                # Determine which species go extinct (each species has its own niche breadth)
                extinction_a = np.abs(env_change_a) > niche_breadth_a
                # Check if all Category A species are extinct
                if extinction_a.all():
                    a_extinct_count += 1
            else:
                # If there are no Category A species, consider it as completely extinct
                a_extinct_count += 1

            # Optional: Simulate survival of Category B species if needed

        # Calculate Probability of Complete Extinction for Category A
        extinction_prob = a_extinct_count / num_simulations
        extinction_probs.append(extinction_prob)

        # Calculate 95% Confidence Interval (Using Wilson Interval)
        lower, upper = proportion_confint(count=a_extinct_count, nobs=num_simulations, alpha=0.05, method='wilson')
        lower_bounds.append(lower)
        upper_bounds.append(upper)

        print(f"Scenario: {scenario}, Category A Occupancy: {a_prop}%, Complete Extinction Probability: {extinction_prob:.4f}, 95% Confidence Interval: [{lower:.4f}, {upper:.4f}]")

        # Add Current Record to csv_data
        csv_data.append({
            'Scenario': scenario,
            'Category A Occupancy (%)': a_prop,
            'Complete Extinction Probability': extinction_prob,
            '95% Confidence Interval Lower': lower,
            '95% Confidence Interval Upper': upper
        })

    # Store Results in Dictionary
    results[scenario]['prob'] = extinction_probs
    results[scenario]['lower'] = lower_bounds
    results[scenario]['upper'] = upper_bounds

# Write Results to CSV File
fieldnames = ['Scenario', 'Category A Occupancy (%)', 'Complete Extinction Probability', '95% Confidence Interval Lower', '95% Confidence Interval Upper']
with open('simulation_results.csv', 'w', newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()  # Write column names
    writer.writerows(csv_data)  # Write data rows

# Plot the Results
plt.figure(figsize=(10,6))

# Background Extinction Scenario
extinction_probs_bg = np.array(results['Background Extinction']['prob'])
lower_bounds_bg = np.array(results['Background Extinction']['lower'])
upper_bounds_bg = np.array(results['Background Extinction']['upper'])

# Plot Probability Curve
plt.plot(a_proportions, extinction_probs_bg, 'o-', color='blue', label='Background Extinction (Small Environmental Changes)')

# Fill Confidence Interval
plt.fill_between(a_proportions, lower_bounds_bg, upper_bounds_bg, color='blue', alpha=0.2)

# Mass Extinction Scenario
extinction_probs_mass = np.array(results['Mass Extinction']['prob'])
lower_bounds_mass = np.array(results['Mass Extinction']['lower'])
upper_bounds_mass = np.array(results['Mass Extinction']['upper'])

# Plot Probability Curve
plt.plot(a_proportions, extinction_probs_mass, 's-', color='red', label='Mass Extinction (Large Environmental Changes)')

# Fill Confidence Interval
plt.fill_between(a_proportions, lower_bounds_mass, upper_bounds_mass, color='red', alpha=0.2)

# Additional Plot Settings
plt.xlabel('Proportion of Niches Occupied by Category A (%)')
plt.ylabel('Probability of Complete Extinction of Category A')
plt.title('Relationship Between Category A Occupancy and Complete Extinction Probability (With 95% Confidence Interval)')
plt.gca().invert_xaxis()  # Invert X-axis to show occupancy proportion from high to low
plt.grid(True)
plt.legend()
plt.savefig('extinction_probability_with_confint_shaded.png')  # Save the plot as a file
plt.show()

# Plot Spatial Distribution of Niches
plt.figure(figsize=(6,6))
plt.scatter(niche_positions[:,0], niche_positions[:,1], c='green', alpha=0.6)
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Random Spatial Distribution of Niches')
plt.grid(True)
plt.savefig('niche_spatial_distribution.png')  # Save the plot as a file
plt.show()