from ucf_crime_recognition.pipeline import ucf_crime_training_flow


if __name__ == "__main__":
    ucf_crime_training_flow.serve(
        name="ucf-crime-training",
        cron="0 2 * * *",
        tags=["ucf-crime", "mlflow", "computer-vision"],
        description="Daily UCF Crime image model training and model selection.",
        parameters={
            "download": False,
            "rebuild_manifest": False,
        },
    )
