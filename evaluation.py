from kfp.v2.dsl import component

# Type annotations for the component artifacts
from kfp.v2.dsl import (
    Input,
    Output,
    Artifact,
    Dataset,
    Model,
    Metrics,
    Markdown,
    ClassificationMetrics,
    OutputPath
)
import os
#credential_path = "C:\\repos\\analysis\\ranathakur123-c51d3b748e05.json"
credential_path = "C:\\repos\\analysis\\test-iris-classification-example.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

@component(
    output_component_file="eval_model.yaml",
    base_image="gcr.io/ranathakur123/test1_ml_image_kfp1/0220901"
)
def evaluation(input_model: Input[Model],
               input_test_set:Input[Dataset],
               output_metrics: Output[Metrics],
               output_pic: OutputPath("pictures"),
               markdown_artifact: Output[Markdown],
               conf_metric: Output[ClassificationMetrics]):
    import pickle
    import pandas as pd
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    import logging
    
    logging.info("evaluation component run started.")
    
    print('1.Read data')
    test_set = pd.read_parquet(input_test_set.path)
    
    X_test=test_set.iloc[:,0:4]
    y_true=test_set.iloc[:,-1]
    
    print('2.Evaluation')
    model = pickle.load(open(input_model.path, 'rb'))
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_true, y_pred)
    
    df=pd.DataFrame(cm)
    #markdown_content=df.read().decode('utf-8')
    #with open(markdown_artifact.path, 'w') as f:
    #        f.write(markdown_content)
    
    df.to_csv(markdown_artifact.path,index=False)
    
    conf_metric.log_confusion_matrix(
        ["Setosa", "Versicolour", "Virginica"],
        confusion_matrix(y_true, y_pred).tolist(),)

    print('3.Generating evaluation metrics')
    accuracy=accuracy_score(y_true, y_pred)
    precision=precision_score(y_true, y_pred, average='macro')
    recall=recall_score(y_true, y_pred, average='macro')
    f1_score=f1_score(y_true, y_pred, average='macro')
    
    
    output_metrics.log_metric("Accuracy=",accuracy)
    output_metrics.log_metric("Precision=",precision)
    output_metrics.log_metric("Recall=",recall)
    output_metrics.log_metric("f1_score=",f1_score)
    
    print('4.Generating plots')
    # prediction dist plot
    ax = pd.Series(y_pred).value_counts().plot.bar()
    ax.grid(True)
    ax.set_title('Predictions');
    graph = ax.get_figure()
    graph.savefig(fname=output_pic)
    
    logging.info("evaluation component compelte.")
    print('SAVED:', output_pic)
    
    logging.info("evaluation component run complete.")