import json
import pathlib

import airflow
import requests
import requests.exceptions as requests_exceptions
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

dag = DAG(
    dag_id="download_rocket_launches",
    start_date=airflow.utils.dates.days_ago(14),
    schedule_interval="@daily",
)

download_launches = BashOperator(
    task_id="download_launches",
    bash_command="curl -o /tmp/launches.json 'https://ll.thespacedevs.com/2.0.0/launch/upcoming/?limit=10&offset=20'",
    dag=dag,
)

def _get_pictures():
	# Ensure directory exists
	pathlib.Path("/tmp/images").mkdir(parents=True, exist_ok=True)

	# Download all pictures in launches.json
	with open("/tmp/launches.json") as f:
		launches = json.load(f)
		image_urls = [launch["image"] for launch in launches["results"]]
		for image_url in image_urls:
			try:
				response = requests.get(image_url)
				image_filename = image_url.split("/")[-1]
				target_file = f"tmp/images/{image_filename}"
				with open(f"/tmp/images/{image_filename}", "wb") as f:
					f.write(response.content)
				print(f"Downloaded {image_url} to {target_file}")
			except requests_exceptions.MissingSchema:
				print(f"{image_url} appears to be an invalid URL.")
			except requests_exceptions.ConnectionError:
				print(f"Could not connect to {image_url}.")
    
get_pictures = PythonOperator(
    task_id="get_pictures",
	python_callable=_get_pictures,
	dag=dag,
)

notify = BashOperator(
    task_id="notify",
	bash_command='echo "There are now $(ls /tmp/images/ | wc -l) images."',
	dag=dag,
)

# 화살표(>>)는 각 테스크 실행 순서를 설정
download_launches >> get_pictures >> notify


# docker run -it -p 8081:8080 -v /home/scar/Desktop/sanghoon/crois/airflow/download_rocket_launchs.py:/opt/airflow/dags/download_rocket_launches.py --entrypoint=/bin/bash --name airflow apache/airflow:latest-python3.8 -c '( \
# airflow db init && \
# airflow users create \
# --username admin \
# --password admin \
# --firstname Anonymous \
# --lastname Admin \
# --role Admin \
# --email admin@example.org \
# ); \
# airflow webserver & \
# airflow scheduler \
# '