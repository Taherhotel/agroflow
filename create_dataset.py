import pandas as pd
import numpy as np

def create_synthetic_dataset(n_samples=1000):
    """
    Create a synthetic dataset for hydroponic fertilizer prediction
    """
    np.random.seed(42)
    
    # Define hydroponic plants and their optimal conditions
    plants = {
        'Lettuce': {
            'ph_range': (5.5, 6.5),
            'tds_range': (560, 840),
            'fertilizer': 'General Hydroponics Flora Series',
            'dosage': 1.2,
            'supplements': ['Cal-Mag Plus', 'FloraNova Grow']
        },
        'Tomato': {
            'ph_range': (5.5, 6.5),
            'tds_range': (1400, 3500),
            'fertilizer': 'Advanced Nutrients pH Perfect',
            'dosage': 1.8,
            'supplements': ['Cal-Mag Plus', 'Big Bud', 'B-52']
        },
        'Cucumber': {
            'ph_range': (5.5, 6.0),
            'tds_range': (1190, 1750),
            'fertilizer': 'General Hydroponics MaxiGro',
            'dosage': 1.5,
            'supplements': ['Cal-Mag Plus', 'Diamond Nectar']
        },
        'Strawberry': {
            'ph_range': (5.5, 6.5),
            'tds_range': (1260, 1540),
            'fertilizer': 'Botanicare Pure Blend Pro',
            'dosage': 1.3,
            'supplements': ['Cal-Mag Plus', 'Sweet Raw']
        },
        'Basil': {
            'ph_range': (5.5, 6.5),
            'tds_range': (700, 1120),
            'fertilizer': 'General Hydroponics FloraMicro',
            'dosage': 1.0,
            'supplements': ['Cal-Mag Plus', 'FloraNova Grow']
        },
        'Spinach': {
            'ph_range': (6.0, 7.0),
            'tds_range': (1260, 1610),
            'fertilizer': 'General Hydroponics MaxiGro',
            'dosage': 1.2,
            'supplements': ['Cal-Mag Plus', 'FloraNova Grow']
        },
        'Bell Pepper': {
            'ph_range': (5.5, 6.5),
            'tds_range': (1400, 3500),
            'fertilizer': 'Advanced Nutrients pH Perfect',
            'dosage': 1.8,
            'supplements': ['Cal-Mag Plus', 'Big Bud', 'B-52']
        },
        'Mint': {
            'ph_range': (5.5, 6.5),
            'tds_range': (560, 840),
            'fertilizer': 'General Hydroponics FloraMicro',
            'dosage': 1.0,
            'supplements': ['Cal-Mag Plus', 'FloraNova Grow']
        }
    }
    
    # Define pH adjustment products
    ph_products = {
        'ph_up': 'General Hydroponics pH Up',
        'ph_down': 'General Hydroponics pH Down'
    }
    
    # Initialize lists for data
    plant_names = []
    ph_values = []
    tds_values = []
    turbidity_values = []
    fertilizers = []
    dosages = []
    supplements = []
    ph_adjustments = []
    
    # Generate data for each plant
    samples_per_plant = n_samples // len(plants)
    for plant, conditions in plants.items():
        for _ in range(samples_per_plant):
            # Generate values within optimal ranges
            ph = np.random.uniform(*conditions['ph_range'])
            tds = np.random.uniform(*conditions['tds_range'])
            turbidity = np.random.uniform(0, 2)  # Low turbidity for hydroponics
            
            # Add some variation to the values
            ph += np.random.normal(0, 0.2)
            tds += np.random.normal(0, 100)
            turbidity += np.random.normal(0, 0.2)
            
            # Clip values to reasonable ranges
            ph = np.clip(ph, 5.0, 7.0)
            tds = np.clip(tds, 500, 3500)
            turbidity = np.clip(turbidity, 0, 5)
            
            # Determine pH adjustment needed
            if ph < 5.5:
                ph_adjustment = ph_products['ph_up']
            elif ph > 6.5:
                ph_adjustment = ph_products['ph_down']
            else:
                ph_adjustment = 'No pH adjustment needed'
            
            # Add some noise to dosage
            dosage = conditions['dosage'] + np.random.normal(0, 0.1)
            dosage = np.clip(dosage, 0.5, 3.0)
            
            # Append values
            plant_names.append(plant)
            ph_values.append(ph)
            tds_values.append(tds)
            turbidity_values.append(turbidity)
            fertilizers.append(conditions['fertilizer'])
            dosages.append(dosage)
            supplements.append(', '.join(conditions['supplements']))
            ph_adjustments.append(ph_adjustment)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Plant': plant_names,
        'pH': ph_values,
        'TDS': tds_values,
        'Turbidity': turbidity_values,
        'Fertilizer': fertilizers,
        'Dosage': dosages,
        'Supplements': supplements,
        'pH_Adjustment': ph_adjustments
    })
    
    return df

def main():
    # Create dataset
    print("Creating synthetic hydroponic dataset...")
    df = create_synthetic_dataset()
    
    # Save to CSV
    df.to_csv('fertilizer_data.csv', index=False)
    print("\nDataset saved as 'fertilizer_data.csv'")
    
    # Display dataset information
    print("\nDataset Info:")
    print(df.info())
    
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nPlant distribution:")
    print(df['Plant'].value_counts())
    
    print("\nFertilizer distribution:")
    print(df['Fertilizer'].value_counts())
    
    print("\nSupplement distribution:")
    print(df['Supplements'].value_counts())
    
    print("\npH Adjustment distribution:")
    print(df['pH_Adjustment'].value_counts())
    
    print("\nDosage statistics by plant:")
    print(df.groupby('Plant')['Dosage'].describe())
    
    print("\nOptimal ranges by plant:")
    for plant in df['Plant'].unique():
        plant_data = df[df['Plant'] == plant]
        print(f"\n{plant}:")
        print(f"pH range: {plant_data['pH'].min():.2f} - {plant_data['pH'].max():.2f}")
        print(f"TDS range: {plant_data['TDS'].min():.2f} - {plant_data['TDS'].max():.2f} ppm")
        print(f"Turbidity range: {plant_data['Turbidity'].min():.2f} - {plant_data['Turbidity'].max():.2f} NTU")
        print(f"Recommended fertilizer: {plant_data['Fertilizer'].iloc[0]}")
        print(f"Recommended supplements: {plant_data['Supplements'].iloc[0]}")
        print(f"Average dosage: {plant_data['Dosage'].mean():.2f} g/L")

if __name__ == "__main__":
    main() 