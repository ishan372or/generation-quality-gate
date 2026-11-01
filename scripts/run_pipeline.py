from pathlib import Path
import sys
import traceback
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_step(name, func):
    print(f"\n{'=' * 70}")
    print(f"STEP: {name}")
    print(f"{'=' * 70}")

    try:
        func()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAILED] {name}")
        print(f"Error: {e}")
        traceback.print_exc()
        raise


def main():
    print("GENERATION QUALITY GATE - ML PIPELINE")
    print(f"Project root: {ROOT}")
    from ml import datagen
    from ml import splits
    from ml import train
    from ml import calibration
    from ml import evaluate
    from ml import bundle
    from ml import features
    run_step(
        "Generate synthetic dataset",
        lambda: datagen.generate_dataset(
            config_path="config/data.yaml"
        ),
    )
    run_step(
        "Create grouped dataset splits",
        lambda: splits.create_splits(
            data_path="artifacts/data/dataset.parquet",
            config_path="config/data.yaml",
        ),
    )
    def train_models():
        train_df = pd.read_csv(
            "artifacts/splits/train.csv"
        )

        train_features, train_labels = (
            features.build_features(train_df)
        )

        models = train.train_all_scorers(
            train_features=train_features,
            train_labels=train_labels,
            config_path="config/train.yaml",
        )

        train.save_models(models)

    run_step(
        "Train models",
        train_models,
    )
    def calibrate_models():
        validation_df = pd.read_csv(
            "artifacts/splits/validation.csv"
        )

        validation_features, validation_labels = (
            features.build_features(validation_df)
        )

        models = evaluate.load_models(
            model_dir="artifacts/models"
        )

        calibrators = calibration.fit_all_calibrators(
            models=models,
            validation_features=validation_features,
            validation_labels=validation_labels,
        )

        for name, calibrator in calibrators.items():
            joblib.dump(
                calibrator,
                f"artifacts/models/{name}_calibrator.joblib",
            )

    run_step(
        "Calibrate models",
        calibrate_models,
    )
    def evaluate_models():
        test_df = pd.read_csv(
            "artifacts/splits/test.csv"
        )

        models = evaluate.load_models(
            model_dir="artifacts/models"
        )

        calibrators = evaluate.load_calibrators(
            calibrator_dir="artifacts/models"
        )

        results = evaluate.evaluate_models(
            test_df=test_df,
            models=models,
            calibrators=calibrators,
            config_path="config/train.yaml",
        )

        evaluate.save_results(
            results,
            output_path="artifacts/evaluation.txt",
        )

        print(
            "\n"
            + evaluate.format_results(results)
        )

    run_step(
        "Evaluate models",
        evaluate_models,
    )
    run_step(
        "Create model bundle",
        lambda: bundle.create_bundle(
            output_dir="artifacts/bundle"
        ),
    )

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()