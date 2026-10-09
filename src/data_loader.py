import pandas as pd
import numpy as np

def load_data():
    """Generates and prepares dataset for text classification."""
    data = {
        "text": [
            "URGENT: Your bank account has been compromised. Verify here http://bit.ly/bank",
            "Hey, are we still meeting for lunch at 1 PM?",
            "Congratulations! You won a $1000 Walmart gift card. Click to claim.",
            "Can you please review the attached quarterly report by EOD?",
            "Your package delivery is pending. Update your details immediately.",
            "Let's schedule the sync call for tomorrow morning.",
            "Claim your lottery winnings now! Send account details.",
            "Thanks for the update. I will check and get back to you.",
            "IRS Notice: You owe unpaid taxes. Call this number immediately.",
            "Don't forget to push your latest code to the development branch."
        ] * 15,  # replicates to form a mini dataset
        "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 15
    }
    df = pd.DataFrame(data)
    # Basic text cleaning
    df["clean_text"] = df["text"].str.lower().str.replace(r"[^\w\s]", "", regex=True)
    return df

if __name__ == "__main__":
    df = load_data()
    print(f"Data loaded successfully: {df.shape[0]} records.")