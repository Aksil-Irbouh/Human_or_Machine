"""Random Forest Training and Evaluation
80% training / 20% Test
Folds = 5
Number of Trees = 100

"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_validate, train_test_split


def train_and_evaluate(X, y, test_size=0.20, cv_folds=5, random_state=42):
    """
    return a dictionnary with :
         - trained model,
         - cross-validation results,
         - confusion Matrix values,
         - features importance values
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(
        f"Training : {len(X_train)} conversations — Test : {len(X_test)} conversations"
    )

    rf = RandomForestClassifier(random_state=random_state)

    # cross validation on the training dataset
    cv = cross_validate(
        rf,
        X_train,
        y_train,
        cv=cv_folds,
        n_jobs=-1,
        scoring=("accuracy", "f1", "roc_auc"),
    )
    print(f"\nCross-validation ({cv_folds} folds, training dataset) :")
    for m in ("accuracy", "f1", "roc_auc"):
        s = cv[f"test_{m}"]
        print(f"  {m:9s} {s.mean():.4f} ± {s.std():.4f}   {np.round(s, 4)}")

    # Evaluation
    rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print("\nEvaluation  :")
    print(classification_report(y_test, y_pred, target_names=["human", "machine"]))
    cm = pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        index=["real human", "real machine"],
        columns=["predicted human", "predicted machine"],
    )
    print("Confusion Matrix :")
    print(cm.to_string())

    imp = pd.Series(rf.feature_importances_, index=X.columns)
    print("\nTop 15 features :")
    print(imp.sort_values(ascending=False).head(15).round(4).to_string())

    return {
        "model": rf,
        "cv": cv,
        "cm": cm,
        "importances": imp,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }
