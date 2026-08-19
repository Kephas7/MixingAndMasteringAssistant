# Model training

Training scripts build synthetic examples, extract features, fit estimators, and write Joblib artifacts into `models/`. Fix the random seed when comparing experiments and record data-generation settings with evaluation results.

Run the EQ and compression training entry points independently. Generated artifacts are intentionally excluded from version control.
