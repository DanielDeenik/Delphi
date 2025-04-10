"""
Simple script to test reading the configuration file.
"""
import json
import os

def load_tracked_stocks():
    """Load the list of tracked stocks from configuration."""
    try:
        with open('config/tracked_stocks.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading tracked stocks: {str(e)}")
        return None

def main():
    """Main function."""
    # Print current working directory
    print(f"Current working directory: {os.getcwd()}")
    
    # Load tracked stocks
    tracked_stocks = load_tracked_stocks()
    
    if tracked_stocks:
        print("Successfully loaded tracked stocks:")
        print(f"Buy stocks ({len(tracked_stocks['buy'])}):")
        for stock in tracked_stocks['buy']:
            print(f"  - {stock}")
        
        print(f"Short stocks ({len(tracked_stocks['short'])}):")
        for stock in tracked_stocks['short']:
            print(f"  - {stock}")
        
        # Print total number of stocks
        total_stocks = len(tracked_stocks['buy']) + len(tracked_stocks['short'])
        print(f"Total number of stocks: {total_stocks}")
    else:
        print("Failed to load tracked stocks")

if __name__ == "__main__":
    main()
