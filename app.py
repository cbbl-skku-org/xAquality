import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os
import warnings

# Suppress serialization and version warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Constants
FEATURES = ["pH", "DO", "COD", "BOD5", "PO4", "NH4", "NO2", "NO3", "Coliform"]
MODEL_PATH = "checkpoints/xAquality_model.pkl"
SCALER_PATH = "checkpoints/scaler_weight.pkl"


# Load artifacts
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


model, scaler = load_artifacts()


def get_wqi_status(wqi):
    if wqi >= 91:
        return "Excellent", "#2ecc71"
    elif wqi >= 76:
        return "Good", "#3498db"
    elif wqi >= 51:
        return "Moderate", "#f1c40f"
    elif wqi >= 26:
        return "Poor", "#e67e22"
    elif wqi >= 10:
        return "Very poor", "#e74c3c"
    else:
        return "Heavily polluted", "#943126"


def predict_wqi(ph, do, cod, bod5, po4, nh4, no2, no3, coliform):
    if model is None or scaler is None:
        return "Error: Model or Scaler not found.", "", ""

    # Create DataFrame
    input_data = pd.DataFrame(
        [[ph, do, cod, bod5, po4, nh4, no2, no3, coliform]], columns=FEATURES
    )

    # Scale and predict
    X_scaled = scaler.transform(input_data.values)
    prediction = model.predict(X_scaled)[0]

    status, color = get_wqi_status(prediction)

    # HTML for beautiful display
    result_html = f"""
    <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: {color}; color: white; font-family: sans-serif;">
        <h2 style="margin: 0;">Predicted WQI</h2>
        <h1 style="font-size: 3em; margin: 10px 0;">{prediction:.2f}</h1>
        <h3 style="margin: 0;">Status: {status}</h3>
    </div>
    """

    return result_html


# Custom CSS for Responsive Grid Layout


# Custom CSS for Responsive Grid Layout
CSS = """
.interactive-grid {
    display: grid !important;
    grid-template-columns: repeat(5, 1fr) !important;
    gap: 15px !important;
    padding: 20px !important;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

/* Flatten Gradio's internal column wrappers to allow 
   individual components to sit in the grid */
.interactive-grid > div {
    display: contents !important;
}

@media (max-width: 1200px) {
    .interactive-grid {
        grid-template-columns: repeat(3, 1fr) !important;
    }
}

@media (max-width: 768px) {
    .interactive-grid {
        grid-template-columns: repeat(2, 1fr) !important;
    }
}

/* Custom CSS Modal */
.modal-box {
    position: fixed !important;
    top: 50% !important;
    left: 50% !important;
    transform: translate(-50%, -50%) !important;
    z-index: 9999 !important;
    background: var(--background-fill-primary) !important;
    padding: 20px !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.5) !important;
    border-radius: 8px !important;
    width: 600px !important;
    max-width: 90% !important;
    border: 1px solid var(--border-color-primary) !important;
    transition: opacity 0.2s ease-in-out;
}

/* Hide the default generic download icon inside the code block */
.modal-box button[aria-label="Download"] {
    display: none !important;
}

.hidden-modal {
    opacity: 0 !important;
    pointer-events: none !important;
    z-index: -9999 !important;
}

.result-card {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    background: rgba(52, 152, 219, 0.15) !important;
    border: 2px solid #3498db !important;
    border-radius: 10px;
    padding: 10px !important;
    text-align: center;
    min-height: 100px;
}

/* Make inputs look consistent */
.interactive-grid .gradio-container {
    min-height: auto !important;
}
"""

# Build UI
with gr.Blocks() as demo:
    gr.Markdown(
        """
    # 💦 xAquality: Water Quality Index Predictor
    """
    )

    with gr.Tabs():
        with gr.Tab("Introduction"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Image("figures/xAquality.png", show_label=False)
                with gr.Column(scale=2):
                    gr.Markdown("### Abstract")
                    try:
                        with open("figures/abstract.txt", "r") as f:
                            abstract_text = f.read()
                    except:
                        abstract_text = "Coming soon..."
                    gr.Markdown(abstract_text)

        with gr.Tab("Single Sample Prediction"):
            gr.Markdown("### 💦 Interactive Parameter Dashboard")

            with gr.Column(elem_classes="interactive-grid"):
                ph = gr.Slider(0, 14, value=7.0, label="🧪 pH")
                do = gr.Number(value=5.0, label="🫧 Dissolved Oxygen (DO) [mg/L]")
                cod = gr.Number(
                    value=15.0, label="⚗️ Chemical Oxygen Demand (COD) [mg/L]"
                )
                bod5 = gr.Number(
                    value=8.0,
                    label="⏳ Biochemical Oxygen Demand over five days (BOD₅) [mg/L]",
                )
                po4 = gr.Number(value=0.05, label="🪴 Phosphate (PO₄³⁻) [mg/L]")
                nh4 = gr.Number(value=0.1, label="🧪 Ammonium (NH₄⁺) [mg/L]")
                no2 = gr.Number(value=0.05, label="🧪 Nitrite (NO₂⁻) [mg/L]")
                no3 = gr.Number(value=2.0, label="🧪 Nitrate (NO₃⁻) [mg/L]")
                coliform = gr.Number(value=5000, label="🦠 Coliform [MPN/100ml]")

                # Result card as a sibling in the grid
                with gr.Column(elem_classes="result-card"):
                    gr.Markdown("#### Final Result")
                    result_output = gr.HTML(
                        "<div style='color: gray; padding: 5px;'>Enter data and click Analyze</div>"
                    )
                    btn = gr.Button("⚡ Analyze", variant="primary")

            btn.click(
                predict_wqi,
                inputs=[ph, do, cod, bod5, po4, nh4, no2, no3, coliform],
                outputs=result_output,
            )

        with gr.Tab("Batch Prediction"):
            gr.Markdown("### Upload CSV for Batch Inference")
            file_input = gr.File(label="Upload Water Quality Data (CSV)")

            # Independent Components (Hidden by default)
            data_preview = gr.Dataframe(
                label="Data Preview", interactive=False, visible=False
            )
            file_output = gr.File(label="Download Predictions", visible=False)
            status_text = gr.Markdown(visible=False)

            def preview_csv_fn(file):
                if file is None:
                    return (
                        gr.Dataframe(visible=False, value=None),
                        gr.File(visible=False, value=None),
                        gr.Markdown(visible=False, value=""),
                    )
                try:
                    file_path = file if isinstance(file, str) else file.name
                    df = pd.read_csv(file_path)
                    return (
                        gr.Dataframe(visible=True, value=df),
                        gr.File(visible=False, value=None),
                        gr.Markdown(visible=False, value=""),
                    )
                except Exception as e:
                    print(f"Preview Error: {e}")
                    return (
                        gr.Dataframe(visible=False, value=None),
                        gr.File(visible=False, value=None),
                        gr.Markdown(visible=False, value=""),
                    )

            def process_csv_fn(file):
                if model is None or scaler is None:
                    return (
                        gr.Dataframe(),
                        gr.File(),
                        gr.Markdown(
                            visible=True, value="Error: Model or Scaler not found."
                        ),
                    )

                try:
                    file_path = file if isinstance(file, str) else file.name
                    df = pd.read_csv(file_path)

                    csv_cols_lower = {c.lower(): c for c in df.columns}
                    missing = [f for f in FEATURES if f.lower() not in csv_cols_lower]

                    if missing:
                        return (
                            gr.Dataframe(),
                            gr.File(),
                            gr.Markdown(
                                visible=True, value=f"Error: Missing columns: {missing}"
                            ),
                        )

                    processed_cols = {
                        f: df[csv_cols_lower[f.lower()]] for f in FEATURES
                    }
                    X = pd.DataFrame(processed_cols)[FEATURES]
                    X_scaled = scaler.transform(X.values)
                    df["Predicted_VN_WQI"] = model.predict(X_scaled)

                    output_path = "predictions.csv"
                    df.to_csv(output_path, index=False)

                    return (
                        gr.Dataframe(visible=True, value=df),
                        gr.File(visible=True, value=output_path),
                        gr.Markdown(visible=True, value="Processing complete!"),
                    )
                except Exception as e:
                    print(f"Processing Error: {e}")
                    return (
                        gr.Dataframe(),
                        gr.File(),
                        gr.Markdown(visible=True, value=f"Error: {str(e)}"),
                    )

            # Preview on upload and clear previous results
            file_input.change(
                fn=preview_csv_fn,
                inputs=file_input,
                outputs=[data_preview, file_output, status_text],
            )

            batch_btn = gr.Button("Run Batch Prediction")
            batch_btn.click(
                process_csv_fn,
                inputs=file_input,
                outputs=[data_preview, file_output, status_text],
            )

    gr.Markdown(
        """
    ---
    ### About the Model
    Coming soon...
        
    **VN_WQI Interpretation (MONRE, 2019):**
    - **91 - 100**: Excellent
    - **76 - 90**: Good
    - **51 - 75**: Moderate
    - **26 - 50**: Poor
    - **10 - 25**: Very poor
    - **< 10**: Heavily polluted
    """
    )

    gr.Markdown(
        """
    ### Citation
    If you find this useful, please cite our work.
    """
    )
    with gr.Row():
        btn_bib = gr.Button("BibTeX")
        btn_endnote = gr.Button("EndNote")
        btn_ris = gr.Button("RIS")

    # Using a CSS modal to create a pop up
    with gr.Column(
        elem_classes=["modal-box", "hidden-modal"], visible=True
    ) as citation_modal:
        citation_text = gr.Code(label="Citation Information", language="markdown")
        with gr.Row():
            download_btn = gr.DownloadButton("Download")
            close_btn = gr.Button("Close")

    def load_citation(fmt):
        import os

        files = {
            "bib": "citations/citation.bib",
            "endnote": "citations/citation.enw",
            "ris": "citations/citation.ris",
        }
        try:
            # Get absolute path relative to this script
            base_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(base_dir, files[fmt])
            with open(file_path, "r") as f:
                content = f.read()
            return gr.update(elem_classes=["modal-box"]), content, file_path
        except Exception as e:
            return (
                gr.update(elem_classes=["modal-box"]),
                f"Coming soon... (Error: {e})",
                None,
            )

    btn_bib.click(
        fn=lambda: load_citation("bib"),
        outputs=[citation_modal, citation_text, download_btn],
    )
    btn_endnote.click(
        fn=lambda: load_citation("endnote"),
        outputs=[citation_modal, citation_text, download_btn],
    )
    btn_ris.click(
        fn=lambda: load_citation("ris"),
        outputs=[citation_modal, citation_text, download_btn],
    )

    close_btn.click(
        fn=lambda: gr.update(elem_classes=["modal-box", "hidden-modal"]),
        outputs=citation_modal,
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=CSS, ssr_mode=False)
