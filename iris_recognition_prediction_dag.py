from datetime import datetime, timedelta
from textwrap import dedent
import time

# import the airflow and its related packages
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.models import XCom


from utils import get_new_iris_image 

default_args = {
    'owner': 'cong',
    'depends_on_past': False,
    'email': ['ch3212@columbia.edu'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(seconds=30),
}

with DAG(
    'iris_recognition_prediction_dag',
    default_args=default_args,
    description='yahoo!',
    schedule_interval=timedelta(days=1),
    start_date=datetime.today().replace(hour=7,minute=0,second=0,microsecond=0),
    catchup=False,
    tags=['yahoo!'],
) as dag:

    # t* examples of tasks created by instantiating operators

    
    t1 = PythonOperator(
        task_id='get new iris image',
        python_callable=get_new_iris_image,
    )

    t2 = BashOperator(
        task_id='train the model',
        bash_command='python train.py',
    )

    t3 = BashOperator(
        task_id='prediction and identification',
        bash_command='python prediction.py',
    )
    
    
    t1 >> t2 >> t3