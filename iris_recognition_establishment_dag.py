from datetime import datetime, timedelta
from textwrap import dedent
import time

# import the airflow and its related packages
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# import the related function
from utils import image_capture
from utils import save_function_to_database


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
    'iris_recognition_establishment_dag',
    default_args=default_args,
    description='iris recognition establishment_dag!',
    schedule_interval=timedelta(days=1),
    start_date=datetime.today().replace(hour=7,minute=0,second=0,microsecond=0),
    catchup=False,
    tags=['yahoo!'],
) as dag:

    # t* examples of tasks created by instantiating operators


    t1 = PythonOperator(
        task_id='capture images',
        python_callable=image_capture,
    )


    t2 = PythonOperator(
        task_id='iris images database',
        python_callable=save_function_to_database,
    )

    t3 = BashOperator(
        task_id='data augmentation and image preprocess',
        bash_command='python preprocess.py',
        retries=1,
    )

    t4 = BashOperator(
        task_id='DL model train and save model',
        bash_command='python train.py',
        retries=1,
    )
    
    t1 >> t2 >> t3 >> t4