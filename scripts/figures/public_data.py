"""Readers for the processed data included in the paper release."""

from pathlib import Path
import json
import sqlite3
from typing import Optional

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"


def load_reference_table() -> pd.DataFrame:
    """Load the 120-motif public table with analysis-compatible column names."""
    public = DATA_DIR / "reference_120.csv"
    if public.exists():
        df = pd.read_csv(public)
    elif (DATA_DIR / "reference_120.csv").exists():
        df = pd.read_csv(DATA_DIR / "reference_120.csv")
    else:
        df = pd.read_csv(
            BASE_DIR / "manuscript" / "Supplementary_Data"
            / "TableS3_reference_dataset_120.csv"
        )
    return df.rename(
        columns={
            "dG_rel_kcal_per_mol": "dG_rel_kcal_per_mol",
            "replicate_SD_kcal_per_mol": "replicate_sd",
        }
    )


def load_logo_metrics() -> pd.DataFrame:
    """Load final strict/few-shot external metrics."""
    public = DATA_DIR / "logo_metrics.csv"
    if public.exists():
        return pd.read_csv(public)
    return pd.read_csv(BASE_DIR / "scripts" / "unknown" / "appendix" / "logo_metrics.csv")


def load_logo_external_predictions() -> pd.DataFrame:
    """Load long-format per-motif LOGO predictions."""
    public = DATA_DIR / "logo_external_predictions.csv"
    if public.exists():
        return pd.read_csv(public)

    head = pd.read_csv(DATA_DIR / "reference_120.csv")
    names = head["Name"].astype(str).tolist()
    rows = []
    result_glob = "scripts/" + "results_logo_*/fold_*/retrained/predictions.json"
    for prediction in sorted(BASE_DIR.glob(result_glob)):
        setting_dir = next(part for part in prediction.parts if part.startswith("results_logo_"))
        setting = setting_dir.removeprefix("results_logo_")
        fold_dir = prediction.parent.parent
        fold = fold_dir.name.removeprefix("fold_")
        payload = json.loads(prediction.read_text())
        metadata = fold_dir / "fold_metadata.json"
        indices = json.loads(metadata.read_text())["test_indices"]
        for index, true_value, predicted in zip(
            indices, payload["test"]["y_true"], payload["test"]["y_pred"]
        ):
            rows.append({
                "setting": setting,
                "held_out_class": fold,
                "motif": names[int(index)],
                "y_true": true_value,
                "y_pred": predicted,
                "signed_error": predicted - true_value,
            })
    if not rows:
        raise FileNotFoundError("No public or source LOGO predictions found")
    return pd.DataFrame(rows)


# Columns of the public (reader-facing) Data 1 schema.
PUBLIC_SCREENING_COLUMNS = (
    "setting", "held_out_class", "screening_stage", "hpo_stage",
    "representation", "model", "hpo_params",
    "internal_validation_pearson_r", "initial_screening_cv_pearson_r_by_fold",
)


def _validate_public_schema(frame: pd.DataFrame, source: str) -> None:
    """Check a Data 1 frame carries the exact public reader-facing schema.

    ``source`` names the path for the error message. Rejects both missing
    public columns and any extra columns, so a hand-edited public CSV or the
    internal full-schema source (bookkeeping columns included) is caught on
    every read path, not only the clean-release fallback.
    """
    present = set(frame.columns)
    expected = set(PUBLIC_SCREENING_COLUMNS)
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    if missing or extra:
        detail = []
        if missing:
            detail.append(f"missing columns: {missing}")
        if extra:
            detail.append(f"unexpected columns: {extra}")
        raise ValueError(
            f"{source} does not carry the public Data 1 schema "
            f"({' and '.join(detail)})"
        )


def load_logo_screening() -> pd.DataFrame:
    """Load the public Supplementary Data 1 screening table.

    Prefers the committed public export (reader-facing schema with renamed
    metric columns). In the clean paper release the manuscript tree is absent,
    so it falls back to ``data/logo_screening_output.csv``. Both paths run the
    same schema check, guarding against silently reading the internal
    full-schema source of the same name in the development tree or a
    hand-edited public file.
    """
    public_path = (
        BASE_DIR / "manuscript" / "Supplementary_Data"
        / "Data1_LOGO_screening_output.csv"
    )
    if public_path.exists():
        frame = pd.read_csv(public_path)
        _validate_public_schema(frame, str(public_path))
        return frame
    fallback = DATA_DIR / "logo_screening_output.csv"
    if not fallback.exists():
        raise FileNotFoundError(
            f"No public Data 1 CSV found at {public_path} or {fallback}"
        )
    frame = pd.read_csv(fallback)
    _validate_public_schema(frame, str(fallback))
    return frame


def load_deployment_screening() -> pd.DataFrame:
    """Load deployment-stage screening results from release or source output."""
    public = DATA_DIR / "deployment_screening.csv"
    if public.exists():
        return pd.read_csv(public)
    return pd.read_csv(
        BASE_DIR / "scripts" / "202609-all120-variousOP" / "run-9.3"
        / "screening_stratified" / "summary_stage1_86reps.csv"
    )


def load_prospective_library() -> pd.DataFrame:
    """Load the frozen 32-candidate prospective library."""
    public = DATA_DIR / "prospective_32.csv"
    if public.exists():
        return pd.read_csv(public)
    return pd.read_csv(BASE_DIR / "manuscript" / "Supplementary_Data" / "TableS7_prospective_library_32.csv")


def load_prospective_pmf() -> pd.DataFrame:
    """Load the eight-candidate prospective PMF validation table."""
    public = DATA_DIR / "prospective_pmf_8.csv"
    if public.exists():
        return pd.read_csv(public)
    return pd.read_csv(BASE_DIR / "manuscript" / "Supplementary_Data" / "prospective_pmf_8.csv")


def load_pmf_summary() -> pd.DataFrame:
    """Load the processed prospective PMF summary."""
    public = DATA_DIR / "pmf_summary.csv"
    if public.exists():
        df = pd.read_csv(public)
    else:
        df = pd.read_csv(BASE_DIR / "scripts" / "202609-all120-variousOP" / "pmf_summary.csv")
    legacy_dg_col = "delta" + "G"
    return df.rename(columns={
        legacy_dg_col: "pmf_dG",
        "dG_rel_kcal_per_mol": "pmf_dG",
        "replicate_sd": "pmf_sd",
    })


def load_ensemble_oof(source: Optional[str] = None) -> pd.DataFrame:
    """Load one section of the combined ensemble OOF export."""
    public = DATA_DIR / "ensemble_oof.csv"
    if public.exists():
        df = pd.read_csv(public)
    else:
        deployment = BASE_DIR / "scripts" / "202609-all120-variousOP" / "run-9.3" / "deployment"
        sources = {
            "oof_predictions": "ensemble_oof_predictions.csv",
            "oof_per_model": "ensemble_oof_per_model.csv",
            "fold_metrics": "ensemble_fold_metrics.csv",
            "best_per_k": "ensemble_exhaustive_best_per_k.csv",
            "selected_members": "ensemble_selected_members.csv",
        }
        frames = []
        for label, filename in sources.items():
            path = deployment / filename
            if path.exists():
                frame = pd.read_csv(path)
                frame.insert(0, "source", label)
                frames.append(frame)
        if not frames:
            raise FileNotFoundError("No public or source ensemble OOF data found")
        df = pd.concat(frames, ignore_index=True)
    if source is not None:
        df = df[df["source"] == source].copy()
    return df


def load_lead_sensitivity() -> pd.DataFrame:
    """Load the SQLite-free lead-sensitivity export."""
    public = DATA_DIR / "lead_sensitivity.csv"
    if public.exists():
        return pd.read_csv(public)
    database = (
        BASE_DIR / "scripts" / "lead_sensitivity" / "lead_sensitivity_results"
        / ("lead_sensitivity_results" + ".db")
    )
    with sqlite3.connect(database) as conn:
        return pd.read_sql_query(
            "SELECT fold_name,lead_count,lead_names,representation_name,model_name,"
            "pearson_r,rmse,mae,r2_score,predictions,true_values "
            "FROM lead_sensitivity_results",
            conn,
        )
