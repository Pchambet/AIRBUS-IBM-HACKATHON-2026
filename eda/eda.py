import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set seaborn style
sns.set_theme(style="whitegrid")

# Paths
data_dir = "/Users/pierre/Desktop/GitHub/AIRBUS-IBM/data"
artifact_dir = "/Users/pierre/.gemini/antigravity/brain/35472bd3-e757-47a4-9509-61378847c3f1"

print("Loading data...")
corr_train = pd.read_csv(os.path.join(data_dir, "corrosions_training.csv"))
env_train = pd.read_csv(os.path.join(data_dir, "environment_training.csv"))

print(f"corrosions_training shape: {corr_train.shape}")
print(f"environment_training shape: {env_train.shape}")

print("\n--- Corrosions Data Summary ---")
print(corr_train.info())
print(corr_train.head())

print("\n--- Environment Data Summary ---")
print(env_train.info())

# Missing values
print("\nMissing values in environment data:")
missing_env = env_train.isnull().sum()
print(missing_env[missing_env > 0])

# Time feature in corrosions
corr_train['observation_date'] = pd.to_datetime(corr_train['observation_date'])
corr_train['year'] = corr_train['observation_date'].dt.year
corr_train['month'] = corr_train['observation_date'].dt.month
corr_train['age_at_obs'] = corr_train['year'] - corr_train['aircraft_delivery_year']

plt.figure(figsize=(10, 6))
sns.histplot(corr_train['age_at_obs'], bins=15, kde=True)
plt.title("Distribution of Aircraft Age at Corrosion Observation")
plt.xlabel("Age (years)")
plt.ylabel("Count")
plt.savefig(os.path.join(artifact_dir, "age_at_corrosion.png"))
plt.close()

plt.figure(figsize=(10, 6))
sns.countplot(data=corr_train, x='year', palette="viridis")
plt.title("Corrosion Observations per Year")
plt.xlabel("Year")
plt.ylabel("Count")
plt.savefig(os.path.join(artifact_dir, "corrosions_per_year.png"))
plt.close()

# Let's create a label for the environment data
# We can identify target = 1 if the year_month for the aircraft exists in corrosions_training
# First, let's extract year_month from observation_date in corrosions_training
corr_train['year_month'] = corr_train['observation_date'].dt.to_period('M').astype(str)

# Create a set of (aircraft_id, year_month) for fast lookup
corrosion_events = set(zip(corr_train['aircraft_id'], corr_train['year_month']))

# Assign label 1 to environment data if it matches a corrosion event
env_train['corrosion_event'] = env_train.apply(
    lambda row: 1 if (row['aircraft_id'], row['year_month']) in corrosion_events else 0, 
    axis=1
)

num_events = env_train['corrosion_event'].sum()
print(f"\nNumber of corrosion events in environment training data: {num_events} out of {len(env_train)}")

# Let's look at correlation matrix of numeric features
# Drop non-numeric and id features
numeric_cols = env_train.select_dtypes(include=[np.number]).columns.tolist()
# Also remove total_parking_minutes if it's too dominant or we just want chemical/weather features
# We'll just take all numeric for now

corr_matrix = env_train[numeric_cols].corr()

# Plot correlation matrix
plt.figure(figsize=(24, 20))
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title("Correlation Matrix of Environmental Features")
plt.tight_layout()
plt.savefig(os.path.join(artifact_dir, "correlation_matrix.png"))
plt.close()

# Calculate correlation with the target variable
target_corr = corr_matrix['corrosion_event'].sort_values(ascending=False)
print("\n--- Correlation with Corrosion Event ---")
print(target_corr.head(10))
print(target_corr.tail(10))

# Plot top 10 positively correlated features against target
top_features = target_corr.index[1:10].tolist() # exclude target itself

plt.figure(figsize=(15, 12))
for i, feature in enumerate(top_features, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x='corrosion_event', y=feature, data=env_train)
    plt.title(f"{feature} vs Corrosion")
plt.tight_layout()
plt.savefig(os.path.join(artifact_dir, "top_features_vs_target.png"))
plt.close()

print("\nEDA script completed successfully. Plots saved to artifacts.")
