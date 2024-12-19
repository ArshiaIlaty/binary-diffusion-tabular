import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data():
    """Prepare and split data for federated learning experiment"""
    # Read original data
    df = pd.read_csv("../data/cervical_train.csv")
    
    # Split into two client datasets
    client1_data, client2_data = train_test_split(
        df, test_size=0.5, random_state=42, stratify=df['Biopsy']
    )
    
    # Save split datasets
    client1_data.to_csv("./data/client1_data.csv", index=False)
    client2_data.to_csv("./data/client2_data.csv", index=False)
   
    print(f"\nData successfully split and saved to:")
    print("- ./data/client1_data.csv")
    print("- ./data/client2_data.csv")
    print(f"Client 1 data size: {len(client1_data)}")
    print(f"Client 2 data size: {len(client2_data)}")

if __name__ == "__main__":
    prepare_data()
