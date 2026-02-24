import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load tactical data
df = pd.read_csv('data/interim/117093/117093_pitch_plane_coordinates.csv')

def generate_report(df):
    # --- STEP 1: FILTER OUT GHOSTS ---
    # Only keep players who appear in at least 60 frames (2 seconds)
    appearance_counts = df['id'].value_counts()
    valid_ids = appearance_counts[appearance_counts >= 60].index
    df = df[df['id'].isin(valid_ids)].copy()

    # --- STEP 2: NORMALIZE (as before) ---
    df['x_metres'] = ((df['x_metres'] - df['x_metres'].min()) / (df['x_metres'].max() - df['x_metres'].min())) * 105
    df['y_metres'] = ((df['y_metres'] - df['y_metres'].min()) / (df['y_metres'].max() - df['y_metres'].min())) * 68
    
    report_data = []
    for player_id in df['id'].unique():
        if player_id == -1: continue
        
        player_df = df[df['id'] == player_id].sort_values('frame')
        
        # Calculate Distance & Speed
        dx = player_df['x_metres'].diff().fillna(0)
        dy = player_df['y_metres'].diff().fillna(0)
        dist = np.sqrt(dx**2 + dy**2).sum()
        
        # Realistic Top Speed cap
        speeds = np.sqrt(dx**2 + dy**2) * 30 * 3.6
        top_speed = min(speeds.max(), 34.5) 
        
        report_data.append({
            'Player_ID': int(player_id),
            'Distance_m': round(dist, 2),
            'Top_Speed_kmh': round(top_speed, 2),
            'Consistency_Score': round((len(player_df) / df['frame'].max()) * 100, 1)
        })

    # return pd.DataFrame(report_data)

    report_df = pd.DataFrame(report_data)
    
    # --- HEATMAP FIX ---
    mvp_id = report_df.loc[report_df['Top_Speed_kmh'].idxmax(), 'Player_ID']
    mvp_df = df[df['id'] == mvp_id]
    
    plt.figure(figsize=(10, 6), facecolor='#1e1e1e')
    ax = plt.gca()
    ax.set_facecolor('#1e1e1e')
    
    # Plotting the heatmap
    sns.kdeplot(x=mvp_df['x_metres'], y=mvp_df['y_metres'], 
                fill=True, cmap='magma', thresh=0, levels=100)
    
    plt.xlim(0, 105)
    plt.ylim(0, 68)
    plt.title(f"Grassroot Talent ID: Positional Heatmap (Player #{mvp_id})", color='white')
    plt.savefig('scout_heatmap_mvp.png')
    plt.close()
    
    return report_df

print("Analyzing Grassroot Talent Data...")
summary = generate_report(df)
print("\n--- SCOUT REPORT SUMMARY ---")
print(summary.sort_values('Top_Speed_kmh', ascending=False).to_string(index=False))
summary.to_csv('final_scout_report.csv', index=False)
print("\nSuccess! Heatmap and CSV report generated.")