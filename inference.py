import pandas as pd
import numpy as np
import joblib
import argparse
import os
import sys

# Constants
FEATURES = ["pH", "DO", "COD", "BOD5", "PO4", "NH4", "NO2", "NO3", "Coliform"]
MODEL_PATH = "checkpoints/xAquality_model.pkl"
SCALER_PATH = "checkpoints/scaler_weight.pkl"


def load_artifacts():
    """Load the model and scaler from disk."""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(SCALER_PATH):
        print(f"Error: Scaler file not found at {SCALER_PATH}")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


def predict(data, model, scaler):
    """
    Scale features and predict WQI.
    data: pd.DataFrame with FEATURES
    """
    # Ensure columns are in the correct order
    data = data[FEATURES]

    # Scale features
    # Note: Check if the scaler was trained on just these features
    X_scaled = scaler.transform(data)

    # Predict
    predictions = model.predict(X_scaled)

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Inference script for Water Quality Index (WQI) prediction."
    )

    # Batch input
    parser.add_argument(
        "--input", type=str, help="Path to a CSV file containing input features."
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save predictions (CSV). Only used with --input.",
    )

    # Single sample input
    parser.add_argument("--ph", type=float, help="pH level")
    parser.add_argument("--do", type=float, help="Dissolved Oxygen (DO)")
    parser.add_argument("--cod", type=float, help="Chemical Oxygen Demand (COD)")
    parser.add_argument("--bod5", type=float, help="Biochemical Oxygen Demand (BOD5)")
    parser.add_argument("--po4", type=float, help="Phosphate (PO4)")
    parser.add_argument("--nh4", type=float, help="Ammonium (NH4)")
    parser.add_argument("--no2", type=float, help="Nitrite (NO2)")
    parser.add_argument("--no3", type=float, help="Nitrate (NO3)")
    parser.add_argument("--coliform", type=float, help="Coliform count")

    args = parser.parse_args()

    # Load model and scaler
    model, scaler = load_artifacts()

    if args.input:
        # Batch inference
        if not os.path.exists(args.input):
            print(f"Error: Input file not found at {args.input}")
            sys.exit(1)

        df = pd.read_csv(args.input)

        # Check if all required features are present
        missing = [f for f in FEATURES if f not in df.columns]
        if missing:
            print(f"Error: Missing features in CSV: {missing}")
            sys.exit(1)

        preds = predict(df, model, scaler)
        df["Predicted_WQI"] = preds

        print("\nBatch Prediction Results:")
        print(df[["Predicted_WQI"]].head())

        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\nResults saved to {args.output}")
        else:
            # Print full results if no output file specified and small enough
            if len(df) <= 20:
                print("\nFull Results:")
                print(df[FEATURES + ["Predicted_WQI"]])
            else:
                print(
                    f"\nPredicted {len(df)} samples. Use --output to save all results."
                )

    elif all(getattr(args, f.lower()) is not None for f in FEATURES):
        # Single sample inference
        sample_data = {f: [getattr(args, f.lower())] for f in FEATURES}
        df_sample = pd.DataFrame(sample_data)

        pred = predict(df_sample, model, scaler)[0]

        print("\nSingle Sample Prediction:")
        for f in FEATURES:
            print(f"{f}: {getattr(args, f.lower())}")
        print("-" * 20)
        print(f"Predicted WQI: {pred:.2f}")

    else:
        print(
            "Error: Please provide either --input <file.csv> or all individual feature values."
        )
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
