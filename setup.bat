@echo off
echo Setting up FL-IDS Enterprise Environment...

REM Create directory structure
mkdir fl-ids-enterprise\kafka 2>nul
mkdir fl-ids-enterprise\preprocessing 2>nul
mkdir fl-ids-enterprise\model 2>nul
mkdir fl-ids-enterprise\federated 2>nul
mkdir fl-ids-enterprise\api 2>nul
mkdir fl-ids-enterprise\monitoring 2>nul
mkdir fl-ids-enterprise\dashboard 2>nul
mkdir fl-ids-enterprise\experiments 2>nul
mkdir fl-ids-enterprise\configs 2>nul
mkdir fl-ids-enterprise\data\CICIDS2018 2>nul
mkdir fl-ids-enterprise\logs 2>nul
mkdir fl-ids-enterprise\models 2>nul
mkdir fl-ids-enterprise\outputs\evaluation 2>nul
mkdir fl-ids-enterprise\outputs\results 2>nul
mkdir fl-ids-enterprise\outputs\dashboard_metrics 2>nul

REM Install Python dependencies
pip install -r requirements.txt

echo Environment setup complete!
echo To start Kafka services: docker-compose up -d
echo To download CICIDS2018 dataset: Please place CSV files in .\data\CICIDS2018\
pause